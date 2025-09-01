#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Optional, List

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose2D
from geometry_msgs.msg import Twist
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

import numpy as np
from casadi import SX, vertcat, horzcat, cos, sin, sqrt, mtimes
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel


# -----------------------
# ACADOS model/solver builder
# -----------------------

def build_acados_solver(params: dict):
    """
    OCP(ACADOS) with:
      - x = [xB, yB, thB, thR1..thR4]  (nx=7)
      - u = Up = [xM, yM, omM, omR1..omR4] (nu=7)
      - p (stage-varying) = [x_ref, y_ref, th_ref, u_last(8), w_chg] (np=12)

    Wheel commands U (8-dim) are computed as algebraic functions of (x,u):
      U = h_u(x, u) = [uL1,uR1, ..., uL4,uR4]
    and used in:
      - cost (effort + stage-0 change penalty)
      - path constraints (box bounds on wheel speeds)
    """
    # ---------- symbols ----------
    M = 4
    nx = 3 + M              # 7
    nu = 3 + M              # 7 (Up only)
    n_u_act = 2 * M         # 8 (wheel cmds computed, not decision vars)
    np_param = 3 + n_u_act + 1  # xref,yref,thref + u_last(8) + w_chg

    # states
    xB, yB, thB = SX.sym('xB'), SX.sym('yB'), SX.sym('thB')
    thR = [SX.sym(f'thR{i+1}') for i in range(M)]
    x = vertcat(xB, yB, thB, *thR)

    # inputs (Up only)
    xM, yM, omM = SX.sym('xM'), SX.sym('yM'), SX.sym('omM')
    omR = [SX.sym(f'omR{i+1}') for i in range(M)]
    u = vertcat(xM, yM, omM, *omR)

    # parameters
    x_ref = SX.sym('x_ref')
    y_ref = SX.sym('y_ref')
    th_ref = SX.sym('th_ref')
    u_last = [SX.sym(f'u_last_{i}') for i in range(n_u_act)]  # 8
    w_chg = SX.sym('w_chg')
    p = vertcat(x_ref, y_ref, th_ref, *u_last, w_chg)

    # ---------- constants ----------
    dt = params['con']['t_delta']
    r_BtoR = params['sys']['r_BtoR']  # list of (x,y)
    robo_dst = params['sys']['robo_dst']
    robo_rdi = params['sys']['robo_rdi']

    Q_trn = params['con']['Q_err_trn']    # 2x2 : translation tracking error weights
    Q_ang = params['con']['Q_err_ang']    # scalar : angle tracking error weight
    Q_hdg = params['con']['Q_hdg']        # scalar : heading alignment error weight
    Q_con = params['con']['Q_con']        # scalar : control effort weight
    Q_chg = params['con']['Q_chg']        # scalar : change penalty weight

    # ---------- dynamics (explicit ODE) ----------
    # r_MtoB = [xB-xM, yB-yM]
    rMx = xB - xM
    rMy = yB - yM
    # vB = omM x r_MtoB -> [-omM*rMy, omM*rMx]
    vBx = -omM * rMy
    vBy =  omM * rMx
    thB_dot = omM
    thR_dot = vertcat(*omR)
    xdot = vertcat(vBx, vBy, thB_dot, thR_dot)

    # ---------- helpers ----------
    def rot2(th):
        # 2x2 rotation matrix
        return vertcat(
            horzcat(cos(th), -sin(th)),
            horzcat(sin(th),  cos(th)),
        )

    Rot = rot2(thB)
    r_MtoB_2 = vertcat(rMx, rMy)

    # ---------- pseudo -> wheel U mapping (as function of x,u) ----------
    # For each robot i:
    # r_MtoR = r_MtoB + Rot * r_BtoR_i
    # vR_pseudo = omM x r_MtoR = [-omM*r_y, omM*r_x]
    # |vR| = sqrt(vx^2 + vy^2)
    # uL = (|vR| - 0.5*omR_i*robo_dst)/robo_rdi
    # uR = (|vR| + 0.5*omR_i*robo_dst)/robo_rdi
    vR_pseudo_list = []
    U_from_pseudo = []  # [uL1,uR1,...]
    for i in range(M):
        r_BtoR_i = vertcat(r_BtoR[i][0], r_BtoR[i][1])
        r_MtoR_i = r_MtoB_2 + mtimes(Rot, r_BtoR_i)
        vRpx = -omM * r_MtoR_i[1]
        vRpy =  omM * r_MtoR_i[0]
        vR_pseudo_list.append(vertcat(vRpx, vRpy))
        vR_norm = sqrt(vRpx*vRpx + vRpy*vRpy + 1e-12)
        uL_i = (vR_norm - 0.5*omR[i]*robo_dst)/robo_rdi
        uR_i = (vR_norm + 0.5*omR[i]*robo_dst)/robo_rdi
        U_from_pseudo += [uL_i, uR_i]
    U_from_pseudo = vertcat(*U_from_pseudo)  # 8-dim

    # ---------- cost ----------
    # tracking (position)
    e_pos = vertcat(xB - x_ref, yB - y_ref)
    J_track_trn = mtimes(mtimes(e_pos.T, SX(Q_trn)), e_pos)
    # tracking (angle) via cos/sin
    e_ang_vec = vertcat(cos(thB) - cos(th_ref), sin(thB) - sin(th_ref))
    J_track_ang = Q_ang * mtimes(e_ang_vec.T, e_ang_vec)
    # heading alignment
    J_hdg = 0
    for i in range(M):
        vR_p = vR_pseudo_list[i]
        vR_p_norm = sqrt(vR_p[0]*vR_p[0] + vR_p[1]*vR_p[1] + 1e-12)
        vR_act = vertcat(vR_p_norm * cos(thR[i]),
                         vR_p_norm * sin(thR[i]))
        dv = vR_act - vR_p
        J_hdg += mtimes(dv.T, dv)
    J_hdg = Q_hdg * J_hdg
    # control effort on wheels (computed)
    J_con = Q_con * mtimes(U_from_pseudo.T, U_from_pseudo)
    # change penalty at stage 0: w_chg in p (0/1)
    U_last_vec = vertcat(*u_last)  # 8
    J_chg0 = Q_chg * w_chg * mtimes((U_from_pseudo - U_last_vec).T, (U_from_pseudo - U_last_vec))
    # total stage cost
    stage_cost = J_track_trn + J_track_ang + J_hdg + J_con + J_chg0

    # ---------- model pack ----------
    model = AcadosModel()
    model.name = 'mpc_cart_multi4_u7'
    model.x = x
    model.u = u
    model.p = p
    model.cost_expr_ext_cost = stage_cost
    model.cost_expr_ext_cost_e = 0

    # ---------- OCP ----------
    ocp = AcadosOcp()
    ocp.model = model

    N = params['con']['n_hor']
    ocp.dims.N = N
    ocp.solver_options.tf = N * dt

    # parameter dimensions & defaults for codegen consistency
    ocp.dims.np = np_param
    ocp.parameter_values = np.zeros((np_param,), dtype=float)

    # costs
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    # dynamics
    ocp.model.f_expl_expr = xdot
    ocp.solver_options.integrator_type = 'ERK'

    # input bounds (on Up only)
    bnd = params['con']['arg_bnd']
    ocp.constraints.lbu = np.full((nu,), -bnd, dtype=float)
    ocp.constraints.ubu = np.full((nu,),  bnd, dtype=float)
    ocp.constraints.idxbu = np.arange(nu, dtype=int)

    # state box (loose)
    x_bnd = params['con']['arg_bnd']
    ocp.constraints.lbx = np.full((nx,), -x_bnd, dtype=float)
    ocp.constraints.ubx = np.full((nx,),  x_bnd, dtype=float)
    ocp.constraints.idxbx = np.arange(nx, dtype=int)

    # wheel command path constraints: u_lower <= U_from_pseudo(x,u) <= u_upper
    ocp.model.con_h_expr = U_from_pseudo
    ocp.constraints.lh = np.full((n_u_act,), params['sys']['u_lower'], dtype=float)
    ocp.constraints.uh = np.full((n_u_act,), params['sys']['u_upper'], dtype=float)

    # initial condition set at runtime (acados expects numpy array)
    ocp.constraints.x0 = np.zeros((nx,), dtype=float)

    # solver options
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.qp_tol = 1e-8
    ocp.solver_options.nlp_tol = 1e-6
    ocp.solver_options.print_level = 2

    # export & create
    ocp.code_export_directory = params['acados']['code_export_dir']
    solver = AcadosOcpSolver(ocp, json_file=f"{ocp.model.name}.json")
    return solver, ocp


# -----------------------
# ROS2 Node
# -----------------------

class MPCControllerNode(Node):
    def __init__(self) -> None:
        super().__init__('mpc_controller_node')

        # ------------------------------------------------------------------
        # %% ############ MULTIAGENT TRANSPORTATION PORJECT: 2025.08.28 #############
        # %% Paremeters and User Defined Values
        # ------------------------------------------------------------------
        # %% 1. System Parameters
        #
        # % 1.1. Cart Parameters
        # params.sys.cart_hgt=1.03;   % Cart height [m]
        # params.sys.cart_wdt=2.04;   % Cart width [m]
        #
        # % 1.2. Robot Parameters
        # params.sys.robo_sze=0.30;    % Radius of modelled robot [m]
        # params.sys.robo_rdi=0.10;    % Robot wheel radius [m]
        # params.sys.robo_dst=0.50;    % Distance between robot wheels [m]
        #
        # % 1.3. Cart and Robot Connection Points (with respect to cart frame)
        # params.sys.n_rbt=4;                             % Number of robots
        # params.sys.r_BtoR(:, 1)=[params.sys.cart_wdt/2; -0.30];
        # params.sys.r_BtoR(:, 2)=[params.sys.cart_wdt/2; +0.20];
        # params.sys.r_BtoR(:, 3)=[+0.20; -params.sys.cart_hgt/2];
        # params.sys.r_BtoR(:, 4)=[-0.60; -params.sys.cart_hgt/2];
        #
        # % 1.4. Bounds Regarding Motor Feasibility
        # params.sys.u_lower=-20;   % [rad/s]
        # params.sys.u_upper=+20;   % [rad/s]
        # ------------------------------------------------------------------
        self.declare_parameter('control_rate_hz', 10.0)
        self.declare_parameter('con.t_delta', 0.1)
        self.declare_parameter('con.n_hor', 20)
        self.declare_parameter('con.arg_bnd', 1e3)

        self.declare_parameter('con.Q_err_trn_x', 100.0)
        self.declare_parameter('con.Q_err_trn_y', 100.0)
        self.declare_parameter('con.Q_err_ang', 1000.0)
        self.declare_parameter('con.Q_hdg', 2000.0)
        self.declare_parameter('con.Q_con', 1e-6)  # 1e-6 / 4
        self.declare_parameter('con.Q_chg', 1e-1)  # 1e-1 / 4

        self.declare_parameter('sys.n_rbt', 4)
        self.declare_parameter('sys.cart_hgt', 1.03)
        self.declare_parameter('sys.cart_wdt', 2.04)
        self.declare_parameter('sys.robo_sze', 0.30)
        self.declare_parameter('sys.robo_dst', 0.50)  # wheelbase
        self.declare_parameter('sys.robo_rdi', 0.10)  # wheel radius

        default_r_BtoR = [
            1.02,  -0.30,
            1.02,   0.20,
            0.20,  -0.515,
           -0.60,  -0.515,
        ]  # x1,y1,...,x4,y4
        self.declare_parameter('sys.r_BtoR', default_r_BtoR)
        self.declare_parameter('sys.u_lower', -40.0)
        self.declare_parameter('sys.u_upper',  40.0)

        self.declare_parameter('acados.code_export_dir', '/tmp/acados_mpc_cart_multi4')

        rate_hz = float(self.get_parameter('control_rate_hz').value)

        r_BtoR_list = self._param_list_to_pairs(self.get_parameter('sys.r_BtoR').value, expected_pairs=4)
        # ------------------------------------------------------------------
        # %% 2. Control Parameters
        #
        # % 2.1. User Defined Initial and Final Pose, Reference Speeds
        # user.ref_spdd=0.5*params.sys.u_upper*params.sys.robo_rdi;
        # (ROS node sets references via /reference/trajectory instead.)
        #
        # % 2.2. Map-Related Paramters
        # map.symbolicInterpolationInterval=0.1;   % Intervals for interpolation
        # map.resolution=10; map.inflation=0.7; map.OriginInWorldCoord=[...]
        #
        # % 2.3. MPC Gains and Misc.
        # params.con.t_delta=0.1;
        # params.con.n_hor=20;
        # params.con.Q_con=1e-6/params.sys.n_rbt;
        # params.con.Q_err_trn=diag([100, 100]);
        # params.con.Q_err_ang=1000;
        # params.con.Q_hdg=2000;
        # params.con.Q_chg=1e-1/params.sys.n_rbt;
        # params.con.arg_bnd=1e+3;
        # params.con.obstacleBuffer=0.05;
        # params.con.prmFactor=0.9;
        # ------------------------------------------------------------------
        self._params = {
            'con': {
                't_delta': float(self.get_parameter('con.t_delta').value),
                'n_hor': int(self.get_parameter('con.n_hor').value),
                'arg_bnd': float(self.get_parameter('con.arg_bnd').value),
                'Q_err_trn': [[float(self.get_parameter('con.Q_err_trn_x').value), 0.0],
                              [0.0, float(self.get_parameter('con.Q_err_trn_y').value)]],
                'Q_err_ang': float(self.get_parameter('con.Q_err_ang').value),
                'Q_hdg': float(self.get_parameter('con.Q_hdg').value),
                'Q_con': float(self.get_parameter('con.Q_con').value),
                'Q_chg': float(self.get_parameter('con.Q_chg').value),
            },
            'sys': {
                'n_rbt': int(self.get_parameter('sys.n_rbt').value),
                'cart_hgt': float(self.get_parameter('sys.cart_hgt').value),
                'cart_wdt': float(self.get_parameter('sys.cart_wdt').value),
                'robo_sze': float(self.get_parameter('sys.robo_sze').value),
                'robo_dst': float(self.get_parameter('sys.robo_dst').value),
                'robo_rdi': float(self.get_parameter('sys.robo_rdi').value),
                'r_BtoR': r_BtoR_list,
                'u_lower': float(self.get_parameter('sys.u_lower').value),
                'u_upper': float(self.get_parameter('sys.u_upper').value),
            },
            'acados': {
                'code_export_dir': self.get_parameter('acados.code_export_dir').value
            }
        }

        # -------- ACADOS solver --------
        self.get_logger().info('Building ACADOS solver (nu=7, wheels as path constraints)...')
        self._solver, self._ocp = build_acados_solver(self._params)
        self._N = self._params['con']['n_hor']

        # -------- subscribers (reference trajectory) --------
        self._sub_path = self.create_subscription(Path, '/reference/trajectory', self._on_path, 10)

        # -------- TF listener (current state from TF) --------
        self.declare_parameter('frame.map', 'odom')
        self.declare_parameter('frame.cart', 'base_link')
        # 부모 프레임 분리: 카트는 world, 로봇은 odom을 기본으로 사용
        self.declare_parameter('frame.cart_parent', 'world')
        self.declare_parameter('frame.robot_parent', 'odom')
        self.declare_parameter('frame.robot_prefix', 'robot')
        self.declare_parameter('frame.robot_suffix', '_base_2')
        self._frame_map = str(self.get_parameter('frame.map').value)
        self._frame_cart = str(self.get_parameter('frame.cart').value)
        self._frame_robot_prefix = str(self.get_parameter('frame.robot_prefix').value)
        self._frame_robot_suffix = str(self.get_parameter('frame.robot_suffix').value)
        self._frame_cart_parent = str(self.get_parameter('frame.cart_parent').value)
        self._frame_robot_parent = str(self.get_parameter('frame.robot_parent').value)

        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=True)

        # -------- publishers (robot-level cmd_vel) --------
        self._pub_cmd_vel = [self.create_publisher(Twist, f'/multirobot/robot{i+1}/cmd_vel', 10) for i in range(4)]

        # -------- debug publishers --------
        self._pub_cart_pose = self.create_publisher(Pose2D, '/cart/pose', 10)
        self._pub_ref_x = self.create_publisher(Float64, '/mpc/xref', 10)
        self._pub_ref_y = self.create_publisher(Float64, '/mpc/yref', 10)
        self._pub_ref_th = self.create_publisher(Float64, '/mpc/thref', 10)
        self._pub_ul_dbg = [self.create_publisher(Float64, f'/debug/robot{i+1}/ul', 10) for i in range(4)]
        self._pub_ur_dbg = [self.create_publisher(Float64, f'/debug/robot{i+1}/ur', 10) for i in range(4)]
        self._pub_om_dbg = [self.create_publisher(Float64, f'/debug/robot{i+1}/omega', 10) for i in range(4)]

        # buffers
        self._xref: Optional[float] = None
        self._yref: Optional[float] = None
        self._thref: Optional[float] = None
        self._path_cache: Optional[Path] = None

        self._xB: Optional[float] = None
        self._yB: Optional[float] = None
        self._thB: Optional[float] = None
        self._thR: List[Optional[float]] = [None]*4

        self._last_U = [0.0]*8  # stage-0 dU penalty

        self._u_init = None
        self._x_init = None

        # log-once flags
        self._warn_ref_printed = False
        self._warn_pose_printed = False
        self._warn_th_printed = False

        # timer
        self._timer = self.create_timer(1.0 / rate_hz, self._control_tick)

        self.get_logger().info(f"MPCControllerNode started. Publishing /robot{{i}}/ul,/ur (i=1..4)")

    # ---------- utils ----------
    @staticmethod
    def _param_list_to_pairs(lst, expected_pairs):
        if len(lst) != 2*expected_pairs:
            raise ValueError(f"sys.r_BtoR must have length {2*expected_pairs} (x1,y1,x2,y2,...)")
        return [(float(lst[2*i]), float(lst[2*i+1])) for i in range(expected_pairs)]

    # ---------- callbacks ----------
    def _on_path(self, msg: Path) -> None:
        # Cache latest path
        self._path_cache = msg

    def _lookup_pose_yaw(self, target_frame: str, parent_frame: Optional[str] = None):
        try:
            parent = parent_frame if parent_frame is not None else self._frame_map
            tf = self._tf_buffer.lookup_transform(parent, target_frame, Time(), timeout=Duration(seconds=0.05))
        except Exception as e:
            raise e
        tx = float(tf.transform.translation.x)
        ty = float(tf.transform.translation.y)
        qx = float(tf.transform.rotation.x)
        qy = float(tf.transform.rotation.y)
        qz = float(tf.transform.rotation.z)
        qw = float(tf.transform.rotation.w)
        yaw = math.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
        return tx, ty, yaw

    # ---------- mapping U(x,u) = wheels from pseudo ----------
    def _compute_wheels_from_pseudo(self, xB, yB, thB, thR, Up):
        # params
        r_BtoR = self._params['sys']['r_BtoR']
        robo_dst = self._params['sys']['robo_dst']
        robo_rdi = self._params['sys']['robo_rdi']

        xM, yM, omM = Up[0], Up[1], Up[2]
        omR = Up[3:7]

        # Rot
        c, s = math.cos(thB), math.sin(thB)
        Rot = ((c, -s),
               (s,  c))

        r_MtoB = (xB - xM, yB - yM)
        U = []
        for i in range(4):
            r_BtoR_i = r_BtoR[i]
            r_MtoR = (r_MtoB[0] + Rot[0][0]*r_BtoR_i[0] + Rot[0][1]*r_BtoR_i[1],
                      r_MtoB[1] + Rot[1][0]*r_BtoR_i[0] + Rot[1][1]*r_BtoR_i[1])
            vRpx = -omM * r_MtoR[1]
            vRpy =  omM * r_MtoR[0]
            vRn = math.sqrt(vRpx*vRpx + vRpy*vRpy) + 1e-8
            uL = (vRn - 0.5*omR[i]*robo_dst)/robo_rdi
            uR = (vRn + 0.5*omR[i]*robo_dst)/robo_rdi
            U += [uL, uR]
        return U  # length 8

    # ---------- publish helpers ----------
    def _publish_cmd_vel_from_wheels(self, U):
        r = self._params['sys']['robo_rdi']
        d = self._params['sys']['robo_dst']
        for i in range(4):
            ul = float(U[2*i])
            ur = float(U[2*i+1])
            v = 0.5 * r * (ur + ul)
            wz = (r / d) * (ur - ul)
            msg = Twist()
            msg.linear.x = v
            msg.linear.y = 0.0
            msg.linear.z = 0.0
            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = wz
            self._pub_cmd_vel[i].publish(msg)

    # ---------- control loop ----------
    def _control_tick(self):
        # readiness checks (log once)
        if self._path_cache is None or len(self._path_cache.poses) == 0:
            if not self._warn_ref_printed:
                self.get_logger().info('Waiting for /reference/trajectory...')
                self._warn_ref_printed = True
            return
        # get current state from TF
        try:
            # 카트는 world -> base_link (기본)
            xB, yB, thB = self._lookup_pose_yaw(self._frame_cart, parent_frame=self._frame_cart_parent)
            thR_list: List[float] = []
            for i in range(4):
                frame_i = f"{self._frame_robot_prefix}{i+1}{self._frame_robot_suffix}"
                # 로봇은 odom -> robot{i}_base_2 (기본)
                _, _, yaw_i = self._lookup_pose_yaw(frame_i, parent_frame=self._frame_robot_parent)
                thR_list.append(yaw_i)
        except Exception:
            if not self._warn_pose_printed:
                self.get_logger().info('Waiting for TFs (map->cart, map->robot{i})...')
                self._warn_pose_printed = True
            return
        # update buffers
        self._xB, self._yB, self._thB = xB, yB, thB
        for i in range(4):
            self._thR[i] = thR_list[i]

        # 1) set initial state equality (x0)
        x0 = [self._xB, self._yB, self._thB, *self._thR]
        x0_arr = np.array(x0, dtype=float)
        self._solver.set(0, 'lbx', x0_arr)
        self._solver.set(0, 'ubx', x0_arr)

        # 2) set per-stage parameters p = [xref, yref, thref, u_last(8), w_chg]
        # Use trajectory points; if fewer than N, repeat last point
        poses = self._path_cache.poses
        # capture k=0 reference for debug publish
        xref0 = yref0 = thref0 = None
        for k in range(self._N):
            idx = k if k < len(poses) else len(poses)-1
            pose_k = poses[idx].pose
            xref_k = float(pose_k.position.x)
            yref_k = float(pose_k.position.y)
            # yaw from quaternion (z,w)
            qw = pose_k.orientation.w
            qz = pose_k.orientation.z
            thref_k = math.atan2(2.0*qz*qw, 1.0 - 2.0*(qz*qz))
            w_chg = 1.0 if k == 0 else 0.0
            p = [xref_k, yref_k, thref_k, *self._last_U, w_chg]
            p_arr = np.array(p, dtype=float)
            self._solver.set(k, 'p', p_arr)
            if k == 0:
                xref0, yref0, thref0 = xref_k, yref_k, thref_k

        # 3) solve
        status = self._solver.solve()
        if status not in (0, "ACADOS_SUCCESS"):
            self.get_logger().warn(f"acados status: {status}")

        # 4) extract first-stage control, compute wheels, and publish cmd_vel
        u0 = self._solver.get(0, 'u')  # Up at stage 0
        U_wheels = self._compute_wheels_from_pseudo(self._xB, self._yB, self._thB, self._thR, u0)
        self._publish_cmd_vel_from_wheels(U_wheels)

        # debug publishes: cart pose, reference(k=0), wheels, omegas
        pose_msg = Pose2D()
        pose_msg.x = float(self._xB)
        pose_msg.y = float(self._yB)
        pose_msg.theta = float(self._thB)
        self._pub_cart_pose.publish(pose_msg)

        if xref0 is not None:
            self._pub_ref_x.publish(Float64(data=float(xref0)))
            self._pub_ref_y.publish(Float64(data=float(yref0)))
            self._pub_ref_th.publish(Float64(data=float(thref0)))

        for i in range(4):
            self._pub_ul_dbg[i].publish(Float64(data=float(U_wheels[2*i])))
            self._pub_ur_dbg[i].publish(Float64(data=float(U_wheels[2*i+1])))
            self._pub_om_dbg[i].publish(Float64(data=float(u0[3+i])))

        self._last_U = U_wheels

        # 5) Set initial guess for next iteration (warm start)
        # Get the optimal trajectory prediction
        x_pred = [self._solver.get(k, 'x') for k in range(self._N + 1)]
        u_pred = [self._solver.get(k, 'u') for k in range(self._N)]

        # Shift the predictions by one step
        self._x_init = x_pred[1:] + [x_pred[-1]] 
        self._u_init = u_pred[1:] + [u_pred[-1]]

        # In the next _control_tick(), BEFORE solver.solve()
        # 3) (Optional but recommended) Warm start
        if self._x_init is not None and self._u_init is not None:
            for k in range(self._N):
                self._solver.set(k, 'x', self._x_init[k])
                self._solver.set(k, 'u', self._u_init[k])
            self._solver.set(self._N, 'x', self._x_init[-1])


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MPCControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
