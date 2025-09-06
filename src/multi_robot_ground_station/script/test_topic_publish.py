import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import ParameterEvent
from std_msgs.msg import Float64
import math

import time
#!/usr/bin/env python3


class SimplePublisherNode(Node):
    def __init__(self):
        super().__init__('simple_publisher_node')

        self.object_pub = self.create_publisher(PoseStamped, '/optitrack/object', 10)
        self.robot1_head_pub = self.create_publisher(PoseStamped, '/optitrack/robot_1_head', 10)
        self.robot2_head_pub = self.create_publisher(PoseStamped, '/optitrack/robot_2_head', 10)
        self.param_event_pub = self.create_publisher(ParameterEvent, '/parameter_events', 10)
        self.robot1_towing_pub = self.create_publisher(Float64, '/robot_1/towing_angle', 10)
        self.robot2_towing_pub = self.create_publisher(Float64, '/robot_2/towing_angle', 10)

        self.timer = self.create_timer(1, self.publish_messages)
        self.begining_time = time.time()

    def publish_messages(self):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'world'
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 1.0
        pose.pose.orientation.w = 1.0
        self.object_pub.publish(pose)

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'world'
        pose.pose.position.x = 1.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 0.5
        rad = math.pi / 10.0
        pose.pose.orientation = self.get_quaternion_from_euler(0, 0, rad)
        self.robot1_head_pub.publish(pose)

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'world'
        pose.pose.position.x = -1.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 0.5
        pose.pose.orientation = self.get_quaternion_from_euler(0, 0, -rad)
        self.robot2_head_pub.publish(pose)


        towing_angle1 = Float64()
        if time.time() - self.begining_time > 5.0:
            towing_angle1.data = math.sin(time.time())
        else:
            towing_angle1.data = 0.0
        self.robot1_towing_pub.publish(towing_angle1)

        towing_angle2 = Float64()
        if time.time() - self.begining_time > 5.0:
            towing_angle2.data = math.cos(time.time())
        else:
            towing_angle2.data = 0.0
        self.robot2_towing_pub.publish(towing_angle2)
    
    def get_quaternion_from_euler(self, roll, pitch, yaw):
        import math
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        from geometry_msgs.msg import Quaternion
        return Quaternion(x=qx, y=qy, z=qz, w=qw)
    

def main(args=None):
    rclpy.init(args=args)
    node = SimplePublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()