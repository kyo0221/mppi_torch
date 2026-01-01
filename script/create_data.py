#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import os
from tf_transformations import euler_from_quaternion


class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')

        self.declare_parameter('output_file', 'data/robot_dynamics_data.npz')
        self.declare_parameter('max_samples', 10000)
        self.declare_parameter('dt_threshold', 0.2)

        output_file = self.get_parameter('output_file').value
        self.max_samples = self.get_parameter('max_samples').value
        self.dt_threshold = self.get_parameter('dt_threshold').value

        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_path = os.path.join(pkg_dir, output_file)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.states = []
        self.actions = []
        self.next_states = []

        self.current_state = None
        self.current_action = None
        self.last_time = None

        self.get_logger().info(f'Data collector initialized. Will save to {self.output_path}')
        self.get_logger().info(f'Max samples: {self.max_samples}')

    def cmd_vel_callback(self, msg):
        self.current_action = np.array([msg.linear.x, msg.angular.z])

    def odom_callback(self, msg):
        current_time = self.get_clock().now()

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        orientation_q = msg.pose.pose.orientation
        quaternion = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, theta = euler_from_quaternion(quaternion)

        new_state = np.array([x, y, theta])

        if self.current_state is not None and self.current_action is not None:
            if self.last_time is not None:
                dt = (current_time - self.last_time).nanoseconds / 1e9

                if dt < self.dt_threshold:
                    self.states.append(self.current_state)
                    self.actions.append(self.current_action)
                    self.next_states.append(new_state)

                    num_samples = len(self.states)
                    if num_samples % 100 == 0:
                        self.get_logger().info(f'Collected {num_samples} samples')

                    if num_samples >= self.max_samples:
                        self.save_data()
                        self.get_logger().info('Max samples reached. Shutting down...')
                        rclpy.shutdown()

        self.current_state = new_state
        self.last_time = current_time

    def save_data(self):
        if len(self.states) == 0:
            self.get_logger().warn('No data to save')
            return

        data = {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'next_states': np.array(self.next_states)
        }

        np.savez(self.output_path, **data)
        self.get_logger().info(f'Saved {len(self.states)} samples to {self.output_path}')

    def destroy_node(self):
        self.save_data()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
