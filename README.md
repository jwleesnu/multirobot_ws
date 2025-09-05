## 멀티로봇 MPC 시뮬레이션 워크스페이스

이 워크스페이스는 3개의 ROS 2 패키지로 구성됩니다.

- reference_generator (C++): 기준 궤적 publish
- mpc_controller_cpp (C++, CasADi/IPOPT): 기준 궤적/TF를 입력받아 로봇 명령 생성
- simulation (C++): cmd_vel을 받아 연속시간 기구학을 적분해 TF publish

### 공통 프레임
- 기준 프레임(부모): `world`(cart), `odom`(robots)
- 카트 프레임(자식): `base_link`
- 로봇 프레임(자식): `robot{i}_base_2` (i = 1..4)

---

## reference_generator
시간 함수 x = 1 − cos(φ), y = sin(φ), yaw = φ + π/2 (φ = t/2)를 샘플링해 Path를 publish 합니다.

- 입력: 없음
- 출력:
  - `/reference/trajectory` (nav_msgs/Path)
    - header.frame_id: `map`
    - poses[k].pose.position.{x,y}
    - poses[k].pose.orientation: yaw만 사용(z,w)
- 동작
  - 노드 시작 시 [0, 3] s 구간 Path 1회 publish
  - 이후 100 Hz로 현재 t부터 3 s horizon Path publish
  - Path 내부 샘플링 간격: 0.10 s (10 Hz)

---

## mpc_controller_cpp
CasADi/IPOPT 기반 MPC로, 기준 궤적(Path)과 TF(cart/robots)를 입력받아 각 로봇의 속도 명령을 publish 합니다.

- 입력
  - `/reference/trajectory` (nav_msgs/Path)
    - k 스테이지별 참조: (x_ref, y_ref, th_ref)
  - TF
    - `world -> base_link` (cart 상태: xB, yB, thB)
    - `odom -> robot{i}_base_2` (로봇 i 헤딩: thRi)
- 출력
  - `/multirobot/robot{i}/cmd_vel` (geometry_msgs/Twist)
    - linear.x = v_i
    - angular.z = ω_i (로봇 헤딩 각속도)
    - 그 외 성분 0
- 디버그 출력(시각화/로깅용)
  - `/cart/pose` (geometry_msgs/Pose2D): xB, yB, thB
  - `/mpc/xref`, `/mpc/yref`, `/mpc/thref` (std_msgs/Float64): k=0 참조
  - `/debug/robot{i}/ul`, `/debug/robot{i}/ur` (std_msgs/Float64): 바퀴 속도
  - `/debug/robot{i}/omega` (std_msgs/Float64): Up의 omRi
  - `/mpc/pseudo_input` (std_msgs/Float64MultiArray): Up(:,0)
- 주요 파라미터(ros2 param)
  - `control_rate_hz` (제어 주기, 기본 10 Hz)
  - con: `t_delta`, `n_hor`, `arg_bnd`, `Q_err_trn_x`, `Q_err_trn_y`, `Q_err_ang`, `Q_hdg`, `Q_con`, `Q_chg`
  - sys: `n_rbt`, `cart_hgt`, `cart_wdt`, `robo_sze`, `robo_dst`(바퀴 간격), `robo_rdi`(바퀴 반경), `r_BtoR`(카트→로봇 연결점), `u_lower`, `u_upper`
  - frame: `map`(기본: odom), `cart`(base_link), `cart_parent`(world), `robot_parent`(odom), `robot_prefix`(robot), `robot_suffix`(_base_2)
- 내부 변수(모델)
  - 상태 x: `[xB, yB, thB, thR1..thR4]`
  - 입력 u(의사입력 Up): `[xM, yM, omM, omR1..omR4]`
  - 일반제약 h(x,u)=U_from_pseudo(x,u)=[uL1,uR1,...,uL4,uR4]

빌드 의존성
- CasADi: Ubuntu에서는 `libcasadi-dev` 설치 권장. 패키지 탐색이 실패하면 `CASADI_PREFIX`/`CASADI_ROOT`/`CONDA_PREFIX` 환경변수로 경로를 제공하세요.

---

## simulation
`/multirobot/robot{i}/cmd_vel`을 받아 디퍼런셜 드라이브 변환으로 바퀴 속도(uL,uR) 계산 후, 연속시간 기구학을 오일러 적분해 TF를 publish 합니다. (400 Hz)

- 입력
  - `/multirobot/robot{i}/cmd_vel` (geometry_msgs/Twist)
    - v_i = linear.x, ω_i = angular.z
- 내부 변환(각 로봇 i)
  - uL = (v_i − 0.5·robo_dst·ω_i) / robo_rdi
  - uR = (v_i + 0.5·robo_dst·ω_i) / robo_rdi
- 출력(TF)
  - `world -> base_link` (cart 포즈)
  - `odom -> robot{i}_base_2` (각 로봇 포즈)
- 파라미터
  - `dt`(기본 0.0025 = 400 Hz), `sys.robo_rdi`, `sys.robo_dst`, `sys.n_rbt`, `sys.r_BtoR`
  - 프레임: `frame.cart_parent`(world), `frame.robot_parent`(odom), `frame.cart`(base_link), `frame.robot_prefix/suffix`

---

## 실행 순서(예시)
1) 환경 소싱
```bash
source /opt/ros/<distro>/setup.bash
```
2) 빌드
```bash
cd /home/jaewoo/ros2_workspace/multirobot_ws
colcon build
source install/setup.bash
```
3) 노드 실행
```bash
ros2 run reference_generator simulation_reference
ros2 run simulation simulation
ros2 run mpc_controller_cpp casadi_controller_node
```

## 시각화/로깅 팁
- RViz2: `/reference/trajectory` Path, TF 트리 표시
- rqt_plot: `/cart/pose/*`, `/mpc/*`, `/debug/robot*/{ul,ur,omega}`, `/multirobot/robot*/cmd_vel/*`
- rosbag2 녹화: `ros2 bag record /reference/trajectory /cart/pose /debug/... /multirobot/...`


