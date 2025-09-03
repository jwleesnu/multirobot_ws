#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import List, Optional, Tuple

import numpy as np
import casadi as ca

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from nav_msgs.msg import Path
from geometry_msgs.msg import Pose2D, Twist
from std_msgs.msg import Float64, Float64MultiArray
from tf2_ros import Buffer, TransformListener


class CasadiMPCControllerNode(Node):
    """ROS2 node implementing the same NMPC problem using CasADi/IPOPT.

    Topics and parameters mirror the existing acados-based controller for drop-in use.
    Obstacle (signed distance) constraint is intentionally omitted per instruction.
    """

    def __init__(self) -> None:
        super().__init__('casadi_mpc_controller_node')

        # ---------------- Parameters (mirror of acados controller) ----------------
        self.declare_parameter('control_rate_hz', 10.0)
        self.declare_parameter('con.t_delta', 0.1)
        self.declare_parameter('con.n_hor', 20)
        self.declare_parameter('con.arg_bnd', 1e3)

        self.declare_parameter('con.Q_err_trn_x', 100.0)
        self.declare_parameter('con.Q_err_trn_y', 100.0)
        self.declare_parameter('con.Q_err_ang', 1000.0)
        self.declare_parameter('con.Q_hdg', 2000.0)
        self.declare_parameter('con.Q_con', 1e-6)
        self.declare_parameter('con.Q_chg', 1e-1)

        self.declare_parameter('sys.n_rbt', 4)
        self.declare_parameter('sys.cart_hgt', 1.03)
        self.declare_parameter('sys.cart_wdt', 2.04)
        self.declare_parameter('sys.robo_sze', 0.30)
        self.declare_parameter('sys.robo_dst', 0.50)
        self.declare_parameter('sys.robo_rdi', 0.10)

        default_r_BtoR = [
            1.02, -0.30,
            1.02,  0.20,
            0.20, -0.515,
           -0.60, -0.515,
        ]
        self.declare_parameter('sys.r_BtoR', default_r_BtoR)
        self.declare_parameter('sys.u_lower', -40.0)
        self.declare_parameter('sys.u_upper',  40.0)

        # Frames
        self.declare_parameter('frame.map', 'odom')
        self.declare_parameter('frame.cart', 'base_link')
        self.declare_parameter('frame.cart_parent', 'world')
        self.declare_parameter('frame.robot_parent', 'odom')
        self.declare_parameter('frame.robot_prefix', 'robot')
        self.declare_parameter('frame.robot_suffix', '_base_2')

        rate_hz = float(self.get_parameter('control_rate_hz').value)

        r_BtoR_list = self._param_list_to_pairs(self.get_parameter('sys.r_BtoR').value, expected_pairs=4)

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
        }

        # Frames
        self._frame_map = str(self.get_parameter('frame.map').value)
        self._frame_cart = str(self.get_parameter('frame.cart').value)
        self._frame_robot_prefix = str(self.get_parameter('frame.robot_prefix').value)
        self._frame_robot_suffix = str(self.get_parameter('frame.robot_suffix').value)
        self._frame_cart_parent = str(self.get_parameter('frame.cart_parent').value)
        self._frame_robot_parent = str(self.get_parameter('frame.robot_parent').value)

        # ---------------- Publishers/Subscribers ----------------
        self._sub_path = self.create_subscription(Path, '/reference/trajectory', self._on_path, 10)

        self._pub_cmd_vel = [self.create_publisher(Twist, f'/multirobot/robot{i+1}/cmd_vel', 10) for i in range(4)]
        self._pub_cart_pose = self.create_publisher(Pose2D, '/cart/pose', 10)
        self._pub_ref_x = self.create_publisher(Float64, '/mpc/xref', 10)
        self._pub_ref_y = self.create_publisher(Float64, '/mpc/yref', 10)
        self._pub_ref_th = self.create_publisher(Float64, '/mpc/thref', 10)
        self._pub_ul_dbg = [self.create_publisher(Float64, f'/debug/robot{i+1}/ul', 10) for i in range(4)]
        self._pub_ur_dbg = [self.create_publisher(Float64, f'/debug/robot{i+1}/ur', 10) for i in range(4)]
        self._pub_om_dbg = [self.create_publisher(Float64, f'/debug/robot{i+1}/omega', 10) for i in range(4)]
        self._pub_up = self.create_publisher(Float64MultiArray, '/mpc/pseudo_input', 10)

        # TF
        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=True)

        # ---------------- CasADi NLP build ----------------
        self.get_logger().info('Building CasADi NLP (IPOPT)...')
        self._nlp = self._build_nlp(self._params)

        # Runtime buffers
        self._path_cache: Optional[Path] = None
        self._xB: Optional[float] = None
        self._yB: Optional[float] = None
        self._thB: Optional[float] = None
        self._thR: List[Optional[float]] = [None] * 4

        self._last_U = [0.0] * (2 * self._nlp['M'])
        self._x_prev: Optional[np.ndarray] = None
        self._lam_x_prev: Optional[np.ndarray] = None
        self._lam_g_prev: Optional[np.ndarray] = None

        self._warn_ref_printed = False
        self._warn_pose_printed = False

        self._timer = self.create_timer(1.0 / rate_hz, self._control_tick)
        self.get_logger().info('CasadiMPCControllerNode started.')

    # ---------------- CasADi model construction ----------------
    def _build_nlp(self, params: dict):
        con = params['con']
        sys = params['sys']

        N: int = int(con['n_hor'])
        M: int = int(sys['n_rbt'])
        NX: int = 3 + M
        NUP: int = 3 + M
        NU: int = 2 * M
        dt: float = float(con['t_delta'])

        Q_trn = np.array(con['Q_err_trn'], dtype=float)
        Q_ang = float(con['Q_err_ang'])
        Q_hdg = float(con['Q_hdg'])
        Q_con = float(con['Q_con'])
        Q_chg = float(con['Q_chg'])

        r_BtoR_pairs: List[Tuple[float, float]] = params['sys']['r_BtoR']
        robo_dst = float(sys['robo_dst'])
        robo_rdi = float(sys['robo_rdi'])

        # Decision variables
        X = ca.SX.sym('X', NX, N + 1)
        Up = ca.SX.sym('Up', NUP, N)
        U = ca.SX.sym('U', NU, N)

        # Parameters
        P_X_curr = ca.SX.sym('P_X_curr', NX, 1)
        P_Xb_desr = ca.SX.sym('P_Xb_desr', 3, N)
        P_U_last = ca.SX.sym('P_U_last', NU, 1)

        def current_to_next(x: ca.SX, up: ca.SX) -> ca.SX:
            posiB = ca.vertcat(x[0], x[1], 0.0)
            thetaB = x[2]
            thetaR = x[3:3 + M]
            posiM = ca.vertcat(up[0], up[1], 0.0)
            omegaM = ca.vertcat(0.0, 0.0, up[2])
            omegaR = up[3:3 + M]
            r_MtoB = posiB - posiM
            # cross(omegaM, r_MtoB)
            v = ca.vertcat(
                omegaM[1] * r_MtoB[2] - omegaM[2] * r_MtoB[1],
                omegaM[2] * r_MtoB[0] - omegaM[0] * r_MtoB[2],
                omegaM[0] * r_MtoB[1] - omegaM[1] * r_MtoB[0],
            )
            posiB_next = posiB + v * dt
            thetaB_next = thetaB + omegaM[2] * dt
            thetaR_next = thetaR + omegaR * dt
            return ca.vertcat(posiB_next[0], posiB_next[1], thetaB_next, thetaR_next)

        def rot2(th: ca.SX) -> ca.SX:
            return ca.vertcat(
                ca.horzcat(ca.cos(th), -ca.sin(th), 0.0),
                ca.horzcat(ca.sin(th),  ca.cos(th), 0.0),
                ca.horzcat(0.0, 0.0, 1.0),
            )

        def pseudo_to_actual_inputs(x: ca.SX, up: ca.SX) -> ca.SX:
            posiB = ca.vertcat(x[0], x[1], 0.0)
            thetaB = x[2]
            posiM = ca.vertcat(up[0], up[1], 0.0)
            omegaM = ca.vertcat(0.0, 0.0, up[2])
            r_MtoB = posiB - posiM
            RotBtoW = rot2(thetaB)
            u_list = []
            for i in range(M):
                rBtoR = ca.vertcat(r_BtoR_pairs[i][0], r_BtoR_pairs[i][1], 0.0)
                r_MtoR = r_MtoB + ca.mtimes(RotBtoW, rBtoR)
                v = ca.vertcat(
                    omegaM[1] * r_MtoR[2] - omegaM[2] * r_MtoR[1],
                    omegaM[2] * r_MtoR[0] - omegaM[0] * r_MtoR[2],
                    omegaM[0] * r_MtoR[1] - omegaM[1] * r_MtoR[0],
                )
                vnorm = ca.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
                omegaR_i = up[3 + i]
                uL = (vnorm - 0.5 * omegaR_i * robo_dst) / robo_rdi
                uR = (vnorm + 0.5 * omegaR_i * robo_dst) / robo_rdi
                u_list += [uL, uR]
            return ca.vertcat(*u_list)

        def heading_cost(x: ca.SX, up: ca.SX) -> ca.SX:
            posiB = ca.vertcat(x[0], x[1], 0.0)
            thetaB = x[2]
            posiM = ca.vertcat(up[0], up[1], 0.0)
            omegaM = ca.vertcat(0.0, 0.0, up[2])
            r_MtoB = posiB - posiM
            RotBtoW = rot2(thetaB)
            total = 0.0
            for i in range(M):
                rBtoR = ca.vertcat(r_BtoR_pairs[i][0], r_BtoR_pairs[i][1], 0.0)
                r_MtoR = r_MtoB + ca.mtimes(RotBtoW, rBtoR)
                v_pseudo = ca.vertcat(
                    omegaM[1] * r_MtoR[2] - omegaM[2] * r_MtoR[1],
                    omegaM[2] * r_MtoR[0] - omegaM[0] * r_MtoR[2],
                    omegaM[0] * r_MtoR[1] - omegaM[1] * r_MtoR[0],
                )
                vnorm = ca.sqrt(v_pseudo[0] * v_pseudo[0] + v_pseudo[1] * v_pseudo[1] + v_pseudo[2] * v_pseudo[2])
                v_actual = ca.vertcat(vnorm * ca.cos(x[3 + i]), vnorm * ca.sin(x[3 + i]), 0.0)
                dvx = v_actual[0] - v_pseudo[0]
                dvy = v_actual[1] - v_pseudo[1]
                total = total + dvx * dvx + dvy * dvy
            return total

        # Objective
        F_cost = 0
        for i in range(N):
            # Tracking (translation)
            e_pos = X[0:2, i + 1] - P_Xb_desr[0:2, i]
            F_cost = F_cost + ca.mtimes([e_pos.T, ca.SX(Q_trn), e_pos])
            # Tracking (angle) using cos/sin
            e_ang_vec = ca.vertcat(ca.cos(X[2, i + 1]) - ca.cos(P_Xb_desr[2, i]),
                                   ca.sin(X[2, i + 1]) - ca.sin(P_Xb_desr[2, i]))
            F_cost = F_cost + Q_ang * ca.mtimes(e_ang_vec.T, e_ang_vec)
            # Heading alignment
            F_cost = F_cost + Q_hdg * heading_cost(X[:, i], Up[:, i])
            # Control effort (mirror MATLAB: U(:,1)'*U(:,i))
            F_cost = F_cost + Q_con * ca.mtimes(U[:, 0].T, U[:, i])
            # Change penalty (du/dt)
            if i == 0:
                du = (U[:, i] - P_U_last) / dt
            else:
                du = (U[:, i] - U[:, i - 1]) / dt
            F_cost = F_cost + Q_chg * ca.mtimes(du.T, du)

        # Constraints
        G_list = []
        # Init
        G_list.append(X[:, 0] - P_X_curr)
        # Dyna, Cont
        for i in range(N):
            G_list.append(current_to_next(X[:, i], Up[:, i]) - X[:, i + 1])
            G_list.append(pseudo_to_actual_inputs(X[:, i], Up[:, i]) - U[:, i])

        g = ca.vertcat(*[ca.reshape(gi, -1, 1) for gi in G_list])

        # Decision vector and parameter vector (column-wise stacking)
        x_vec = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(Up, -1, 1), ca.reshape(U, -1, 1))
        p_vec = ca.vertcat(ca.reshape(P_X_curr, -1, 1), ca.reshape(P_Xb_desr, -1, 1), ca.reshape(P_U_last, -1, 1))

        nlp = {'x': x_vec, 'f': F_cost, 'g': g, 'p': p_vec}

        # IPOPT options
        ipopt_common = {
            'print_time': False,
            'ipopt': {
                'max_iter': 3000,
                'tol': 1e-6,
                'print_level': 3,
            }
        }
        ipopt_init = dict(ipopt_common)
        ipopt_init['ipopt'] = dict(ipopt_common['ipopt'])
        ipopt_init['ipopt']['warm_start_init_point'] = 'no'

        ipopt_warm = dict(ipopt_common)
        ipopt_warm['ipopt'] = dict(ipopt_common['ipopt'])
        ipopt_warm['ipopt']['warm_start_init_point'] = 'yes'
        ipopt_warm['ipopt']['warm_start_bound_frac'] = 1e-16
        ipopt_warm['ipopt']['warm_start_bound_push'] = 1e-16
        ipopt_warm['ipopt']['warm_start_mult_bound_push'] = 1e-16
        ipopt_warm['ipopt']['warm_start_slack_bound_frac'] = 1e-16
        ipopt_warm['ipopt']['warm_start_slack_bound_push'] = 1e-16
        ipopt_warm['ipopt']['max_cpu_time'] = 0.5 * dt

        solver_init = ca.nlpsol('npp_solver_init', 'ipopt', nlp, ipopt_init)
        solver_warm = ca.nlpsol('npp_solver_warm', 'ipopt', nlp, ipopt_warm)

        # Sizes for bounds and reshaping
        lgth = {
            'X_col': (N + 1) * NX,
            'Up_col': N * NUP,
            'U_col': N * NU,
            'P_X_curr_col': NX,
            'P_Xb_desr_col': 3 * N,
            'P_U_last_col': NU,
            'G_Init_col': NX,
            'G_Dyna_col': N * NX,
            'G_Cont_col': N * NU,
        }

        return {
            'solver_init': solver_init,
            'solver_warm': solver_warm,
            'N': N,
            'M': M,
            'NX': NX,
            'NUP': NUP,
            'NU': NU,
            'dt': dt,
            'lgth': lgth,
        }

    # ---------------- Timer callback ----------------
    def _control_tick(self) -> None:
        if self._path_cache is None or len(self._path_cache.poses) == 0:
            if not self._warn_ref_printed:
                self.get_logger().info('Waiting for /reference/trajectory...')
                self._warn_ref_printed = True
            return

        # Get current state via TF
        try:
            xB, yB, thB = self._lookup_pose_yaw(self._frame_cart, parent_frame=self._frame_cart_parent)
            thR_list: List[float] = []
            for i in range(4):
                frame_i = f"{self._frame_robot_prefix}{i+1}{self._frame_robot_suffix}"
                _, _, yaw_i = self._lookup_pose_yaw(frame_i, parent_frame=self._frame_robot_parent)
                thR_list.append(yaw_i)
        except Exception:
            if not self._warn_pose_printed:
                self.get_logger().info('Waiting for TFs (map->cart, map->robot{i})...')
                self._warn_pose_printed = True
            return

        self._xB, self._yB, self._thB = xB, yB, thB
        for i in range(4):
            self._thR[i] = thR_list[i]

        # Build parameter vector p = [X_curr; Xb_desr(3N); U_last]
        N = self._nlp['N']
        NX = self._nlp['NX']
        NU = self._nlp['NU']
        dt = self._nlp['dt']

        poses = self._path_cache.poses
        xref = np.zeros((N,), dtype=float)
        yref = np.zeros((N,), dtype=float)
        thref = np.zeros((N,), dtype=float)
        for k in range(N):
            idx = k if k < len(poses) else len(poses) - 1
            pose_k = poses[idx].pose
            xref[k] = float(pose_k.position.x)
            yref[k] = float(pose_k.position.y)
            qw = pose_k.orientation.w
            qz = pose_k.orientation.z
            thref[k] = math.atan2(2.0 * qz * qw, 1.0 - 2.0 * (qz * qz))

        P = np.zeros((NX + 3 * N + NU,), dtype=float)
        # X_curr
        P[0:3] = [self._xB, self._yB, self._thB]
        P[3:3 + 4] = self._thR  # thetaR1..4
        # Xb_desr stacked column-wise [x;y;theta] per stage
        base = NX
        for i in range(N):
            P[base + 3 * i + 0] = xref[i]
            P[base + 3 * i + 1] = yref[i]
            P[base + 3 * i + 2] = thref[i]
        base = NX + 3 * N
        P[base:base + NU] = np.array(self._last_U, dtype=float)

        # Bounds on variables
        arg_bnd = float(self._params['con']['arg_bnd'])
        lbx_x = -arg_bnd * np.ones((self._nlp['lgth']['X_col'],), dtype=float)
        ubx_x = +arg_bnd * np.ones_like(lbx_x)
        lbx_up = -arg_bnd * np.ones((self._nlp['lgth']['Up_col'],), dtype=float)
        ubx_up = +arg_bnd * np.ones_like(lbx_up)
        lbx_u = float(self._params['sys']['u_lower']) * np.ones((self._nlp['lgth']['U_col'],), dtype=float)
        ubx_u = float(self._params['sys']['u_upper']) * np.ones_like(lbx_u)

        lbx = np.concatenate([lbx_x, lbx_up, lbx_u])
        ubx = np.concatenate([ubx_x, ubx_up, ubx_u])

        # Bounds on constraints (Init, Dyna, Cont) — all equalities => zeros
        lbg = np.zeros((self._nlp['lgth']['G_Init_col'] + self._nlp['lgth']['G_Dyna_col'] + self._nlp['lgth']['G_Cont_col'],), dtype=float)
        ubg = np.zeros_like(lbg)

        # Initial guess
        if self._x_prev is None:
            X0 = np.tile(np.array([self._xB, self._yB, self._thB, *self._thR], dtype=float), (self._nlp['N'] + 1, 1)).T
            Up0 = np.tile(np.array([self._xB, self._yB, 1e-3, 0.0, 0.0, 0.0, 0.0], dtype=float), (self._nlp['N'], 1)).T
            U0 = np.zeros((self._nlp['NU'], self._nlp['N']), dtype=float)
            x0 = np.concatenate([
                X0.reshape((-1,), order='F'),
                Up0.reshape((-1,), order='F'),
                U0.reshape((-1,), order='F')
            ])
            solver = self._nlp['solver_init']
            arg = {'x0': x0, 'p': P, 'lbx': lbx, 'ubx': ubx, 'lbg': lbg, 'ubg': ubg}
        else:
            solver = self._nlp['solver_warm']
            arg = {
                'x0': self._x_prev,
                'p': P,
                'lbx': lbx,
                'ubx': ubx,
                'lbg': lbg,
                'ubg': ubg,
                'lam_x0': self._lam_x_prev if self._lam_x_prev is not None else np.zeros_like(self._x_prev),
                'lam_g0': self._lam_g_prev if self._lam_g_prev is not None else np.zeros_like(lbg),
            }

        try:
            res = solver(**arg)
        except Exception as e:
            self.get_logger().warn(f'IPOPT failed: {e}')
            return

        x_opt = np.array(res['x']).reshape((-1,))
        lam_x_opt = np.array(res['lam_x']).reshape((-1,)) if 'lam_x' in res else None
        lam_g_opt = np.array(res['lam_g']).reshape((-1,)) if 'lam_g' in res else None

        self._x_prev = x_opt
        self._lam_x_prev = lam_x_opt
        self._lam_g_prev = lam_g_opt

        # Unstack to get Up(:,0) and U(:,0)
        nX = self._nlp['lgth']['X_col']
        nUp = self._nlp['lgth']['Up_col']
        NX = self._nlp['NX']
        N = self._nlp['N']
        NU = self._nlp['NU']

        vec_X = x_opt[0:nX]
        vec_Up = x_opt[nX:nX + nUp]
        vec_U = x_opt[nX + nUp:]

        Up_mat = vec_Up.reshape((self._nlp['NUP'], N), order='F')
        U_mat = vec_U.reshape((NU, N), order='F')

        up0 = Up_mat[:, 0]
        u0 = U_mat[:, 0]

        # Publish cmd_vel from wheel speeds u0
        self._publish_cmd_vel_from_wheels(u0.tolist())

        # Publish debug topics
        up_msg = Float64MultiArray()
        up_msg.data = [float(val) for val in up0]
        self._pub_up.publish(up_msg)

        pose_msg = Pose2D()
        pose_msg.x = float(self._xB)
        pose_msg.y = float(self._yB)
        pose_msg.theta = float(self._thB)
        self._pub_cart_pose.publish(pose_msg)

        self._pub_ref_x.publish(Float64(data=float(xref[0])))
        self._pub_ref_y.publish(Float64(data=float(yref[0])))
        self._pub_ref_th.publish(Float64(data=float(thref[0])))

        for i in range(4):
            self._pub_ul_dbg[i].publish(Float64(data=float(u0[2 * i + 0])))
            self._pub_ur_dbg[i].publish(Float64(data=float(u0[2 * i + 1])))
            self._pub_om_dbg[i].publish(Float64(data=float(up0[3 + i])))

        # Store for next iteration
        self._last_U = u0.tolist()

    # ---------------- Utilities ----------------
    @staticmethod
    def _param_list_to_pairs(lst, expected_pairs):
        if len(lst) != 2 * expected_pairs:
            raise ValueError(f"sys.r_BtoR must have length {2 * expected_pairs} (x1,y1,x2,y2,...)")
        return [(float(lst[2 * i]), float(lst[2 * i + 1])) for i in range(expected_pairs)]

    def _on_path(self, msg: Path) -> None:
        self._path_cache = msg

    def _lookup_pose_yaw(self, target_frame: str, parent_frame: Optional[str] = None):
        parent = parent_frame if parent_frame is not None else self._frame_map
        tf = self._tf_buffer.lookup_transform(parent, target_frame, Time(), timeout=Duration(seconds=0.05))
        tx = float(tf.transform.translation.x)
        ty = float(tf.transform.translation.y)
        qx = float(tf.transform.rotation.x)
        qy = float(tf.transform.rotation.y)
        qz = float(tf.transform.rotation.z)
        qw = float(tf.transform.rotation.w)
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return tx, ty, yaw

    def _publish_cmd_vel_from_wheels(self, U: List[float]) -> None:
        r = self._params['sys']['robo_rdi']
        d = self._params['sys']['robo_dst']
        for i in range(4):
            ul = float(U[2 * i])
            ur = float(U[2 * i + 1])
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


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CasadiMPCControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


