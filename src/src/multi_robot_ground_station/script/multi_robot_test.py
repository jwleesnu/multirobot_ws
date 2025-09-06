import numpy as np
import matplotlib.pyplot as plt

linear_vel = 1.0 # m/s
angular_vel = 0.5 # rad/s
radius = 1.0 # m

t = np.linspace(0, 10, 1000)

phase_0 = 0
phase_1 = np.pi/3
phase_2 = 2*np.pi/3


agent1_pose_xy = np.array([radius * np.cos(angular_vel * t + phase_0) + linear_vel * t,
                           radius * np.sin(angular_vel * t + phase_0)]).T 
agent2_pose_xy = np.array([radius * np.cos(angular_vel * t + phase_1) + linear_vel * t,
                            radius * np.sin(angular_vel * t + phase_1)]).T
agent3_pose_xy = np.array([radius * np.cos(angular_vel * t + phase_2) + linear_vel * t,
                            radius * np.sin(angular_vel * t + phase_2)]).T
plt.figure()
plt.axis('equal')
plt.grid()
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Multi-Robot Circular Trajectories')

plt.plot(agent1_pose_xy[:,0], agent1_pose_xy[:,1], label='Agent 1')
plt.plot(agent2_pose_xy[:,0], agent2_pose_xy[:,1], label='Agent 2')
plt.plot(agent3_pose_xy[:,0], agent3_pose_xy[:,1], label='Agent 3')

# agent가 같이 있는 원 표시
for i in range(4):
    if i == 0:
        continue
    circle = plt.Circle((linear_vel * t[i*300], 0), radius, color= 'gray', fill=False, linestyle='--', label='Formation Circle')
    plt.scatter(linear_vel * t[i*300], 0, color='black') # 원의 중심점
    plt.scatter(agent1_pose_xy[i*300,0], agent1_pose_xy[i*300,1], color='blue') # agent 1 위치
    plt.scatter(agent2_pose_xy[i*300,0], agent2_pose_xy[i*300,1], color='orange') # agent 2 위치
    plt.scatter(agent3_pose_xy[i*300,0], agent3_pose_xy[i*300,1], color='green') # agent 3 위치
    plt.gca().add_artist(circle)
    # 경로의 tangent 선 표시
    agent1_dir = np.diff(agent1_pose_xy[i*300-1:i*300+2], axis=0)
    normal_vec1 = np.array([-agent1_dir[0,1], agent1_dir[0,0]])
    agent2_dir = np.diff(agent2_pose_xy[i*300-1:i*300+2], axis=0)
    normal_vec2 = np.array([-agent2_dir[0,1], agent2_dir[0,0]])
    agent3_dir = np.diff(agent3_pose_xy[i*300-1:i*300+2], axis=0)
    normal_vec3 = np.array([-agent3_dir[0,1], agent3_dir[0,0]])
    normal_vec1 = normal_vec1 / np.linalg.norm(normal_vec1) * 0.5
    normal_vec2 = normal_vec2 / np.linalg.norm(normal_vec2) * 0.5
    normal_vec3 = normal_vec3 / np.linalg.norm(normal_vec3) * 0.5
    # draw dottedd line
    line_start_1 = agent1_pose_xy[i*300] - normal_vec1 * 1000
    line_end_1 = agent1_pose_xy[i*300] + normal_vec1 * 1000
    plt.plot([line_start_1[0], line_end_1[0]], [line_start_1[1], line_end_1[1]], color='blue', linestyle='--', alpha=0.5)
    line_start_2 = agent2_pose_xy[i*300] - normal_vec2 * 1000
    line_end_2 = agent2_pose_xy[i*300] + normal_vec2 * 1000
    plt.plot([line_start_2[0], line_end_2[0]], [line_start_2[1], line_end_2[1]], color='orange', linestyle='--', alpha=0.5)
    line_start_3 = agent3_pose_xy[i*300] - normal_vec3 * 1000
    line_end_3 = agent3_pose_xy[i*300] + normal_vec3 * 1000
    plt.plot([line_start_3[0], line_end_3[0]], [line_start_3[1], line_end_3[1]], color='green', linestyle='--', alpha=0.5)
plt.show()