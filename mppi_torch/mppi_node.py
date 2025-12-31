#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float64MultiArray
import torch
import torch.nn as nn
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory


class DynamicsModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3):
        super(DynamicsModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MPPIController:
    def __init__(self, dynamics_model, horizon, num_samples, lambda_weight, dt, device):
        self.dynamics_model = dynamics_model
        self.horizon = horizon
        self.num_samples = num_samples
        self.lambda_weight = lambda_weight
        self.dt = dt
        self.device = device
        self.u_prev = torch.zeros(horizon, 2, device=device)

    def compute_control(self, state, reference_path, obstacles):
        noise = torch.randn(self.num_samples, self.horizon, 2, device=self.device)
        u_samples = self.u_prev.unsqueeze(0) + noise

        costs = torch.zeros(self.num_samples, device=self.device)

        for i in range(self.num_samples):
            state_pred = state.clone()
            for t in range(self.horizon):
                u = u_samples[i, t]

                model_input = torch.cat([state_pred, u])
                state_delta = self.dynamics_model(model_input.unsqueeze(0)).squeeze(0)
                state_pred = state_pred + state_delta

                if len(reference_path) > t:
                    ref_pos = reference_path[min(t, len(reference_path)-1)]
                    path_cost = torch.sum((state_pred[:2] - ref_pos) ** 2)
                else:
                    path_cost = 0.0

                obstacle_cost = 0.0
                for obs in obstacles:
                    dist = torch.norm(state_pred[:2] - obs)
                    if dist < 0.5:
                        obstacle_cost += 10.0 / (dist + 0.1)

                control_cost = torch.sum(u ** 2) * 0.01

                costs[i] += path_cost + obstacle_cost + control_cost

        weights = torch.exp(-self.lambda_weight * costs)
        weights = weights / torch.sum(weights)

        u_opt = torch.sum(weights.unsqueeze(1).unsqueeze(2) * u_samples, dim=0)
        self.u_prev = u_opt

        return u_opt[0]


class MPPINode(Node):
    def __init__(self):
        super().__init__('mppi_node')

        self.declare_parameter('horizon', 20)
        self.declare_parameter('num_samples', 1000)
        self.declare_parameter('lambda_weight', 1.0)
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('model_path', 'weights/dynamics_model.pth')
        self.declare_parameter('control_frequency', 10.0)

        self.horizon = self.get_parameter('horizon').value
        self.num_samples = self.get_parameter('num_samples').value
        self.lambda_weight = self.get_parameter('lambda_weight').value
        self.dt = self.get_parameter('dt').value
        model_path = self.get_parameter('model_path').value
        control_freq = self.get_parameter('control_frequency').value

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dynamics_model = DynamicsModel().to(self.device)
        pkg_share = get_package_share_directory('mppi_torch')
        full_model_path = os.path.join(pkg_share, model_path)

        if os.path.exists(full_model_path):
            self.dynamics_model.load_state_dict(torch.load(full_model_path, map_location=self.device))
            self.get_logger().info(f'Loaded model from {full_model_path}')
        else:
            self.get_logger().warn(f'Model file not found at {full_model_path}, using untrained model')

        self.dynamics_model.eval()

        self.mppi = MPPIController(
            self.dynamics_model,
            self.horizon,
            self.num_samples,
            self.lambda_weight,
            self.dt,
            self.device
        )

        self.path_sub = self.create_subscription(
            Path,
            '/e2e_planner/path',
            self.path_callback,
            10
        )

        self.obstacle_sub = self.create_subscription(
            Float64MultiArray,
            '/obstacle/pos',
            self.obstacle_callback,
            10
        )

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.current_path = []
        self.obstacles = []
        self.current_state = torch.zeros(3, device=self.device)

        self.timer = self.create_timer(1.0 / control_freq, self.control_loop)

        self.get_logger().info('MPPI Node initialized')

    def path_callback(self, msg):
        self.current_path = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            self.current_path.append(torch.tensor([x, y], device=self.device))

    def obstacle_callback(self, msg):
        self.obstacles = []
        data = np.array(msg.data)
        num_obstacles = len(data) // 3
        for i in range(num_obstacles):
            x = data[i * 3]
            y = data[i * 3 + 1]
            self.obstacles.append(torch.tensor([x, y], device=self.device))

    def control_loop(self):
        if len(self.current_path) == 0:
            return

        u_opt = self.mppi.compute_control(
            self.current_state,
            self.current_path,
            self.obstacles
        )

        cmd = Twist()
        cmd.linear.x = float(u_opt[0])
        cmd.angular.z = float(u_opt[1])
        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = MPPINode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
