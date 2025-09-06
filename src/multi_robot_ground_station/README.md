## How to use
```
cd ~/ros2_ws/src
git clone https://github.com/tmddn833/multi_robot_ground_station.git
mv multi_robot_ground_station robot_ground_station

cd ..
colcon build  --symlink-install
source ~/.bashrc

ros2 launch ground_station optitrack_calib_launch.py
```
