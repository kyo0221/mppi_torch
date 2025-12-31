#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
import torch
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory
from torchrl.envs import ModelBasedEnvBase
from torchrl.modules import MPPIPlanner
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from tensordict import TensorDict


class RobotDynamicsEnv(ModelBasedEnvBase):
    def __init__(self, dynamics_model, device):
        super().__init__(device=device)
        self.dynamics_model = dynamics_model

        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(shape=(3,), device=device)
        )

        self.action_spec = BoundedTensorSpec(
            low=torch.tensor([-2.0, -0.5], device=device),
            high=torch.tensor([2.0, 0.5], device=device),
            shape=(2,),
            device=device
        )

        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,), device=device)

        self.reference_path = []
        self.obstacles = []

    def _step(self, tensordict):
        state = tensordict["observation"]
        action = tensordict["action"]

        model_input = torch.cat([state, action], dim=-1)
        state_delta = self.dynamics_model(model_input)
        next_state = state + state_delta

        reward = self._compute_reward(next_state, action)

        done = torch.zeros(next_state.shape[:-1] + (1,), device=self.device, dtype=torch.bool)

        out = TensorDict({
            "observation": next_state,
            "reward": reward,
            "done": done,
        }, batch_size=tensordict.batch_size)

        return out

    def _reset(self, tensordict=None):
        if tensordict is None or "observation" not in tensordict:
            observation = torch.zeros(3, device=self.device)
        else:
            observation = tensordict["observation"]

        out = TensorDict({
            "observation": observation,
        }, batch_size=[])

        return out

    def _compute_reward(self, state, action):
        reward = torch.zeros(state.shape[:-1] + (1,), device=self.device)

        if len(self.reference_path) > 0:
            distances = torch.stack([torch.norm(state[..., :2] - ref) for ref in self.reference_path])
            min_distance = torch.min(distances, dim=0)[0]
            path_cost = min_distance ** 2
            reward = reward - path_cost.unsqueeze(-1)

        if len(self.obstacles) > 0:
            for obs in self.obstacles:
                dist = torch.norm(state[..., :2] - obs, dim=-1)
                obstacle_cost = 10.0 / (dist + 0.1)
                obstacle_cost = torch.where(dist < 0.5, obstacle_cost, torch.zeros_like(obstacle_cost))
                reward = reward - obstacle_cost.unsqueeze(-1)

        control_cost = torch.sum(action ** 2, dim=-1, keepdim=True) * 0.01
        reward = reward - control_cost

        return reward

    def set_reference_path(self, path):
        self.reference_path = path

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles


class MPPINode(Node):
    def __init__(self):
        super().__init__('mppi_node')

        self.declare_parameter('planning_horizon', 20)
        self.declare_parameter('optim_steps', 10)
        self.declare_parameter('num_candidates', 1000)
        self.declare_parameter('top_k', 100)
        self.declare_parameter('model_path', 'weights/dynamics_model.pt')
        self.declare_parameter('control_frequency', 10.0)

        planning_horizon = self.get_parameter('planning_horizon').value
        optim_steps = self.get_parameter('optim_steps').value
        num_candidates = self.get_parameter('num_candidates').value
        top_k = self.get_parameter('top_k').value
        model_path = self.get_parameter('model_path').value
        control_freq = self.get_parameter('control_frequency').value

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        pkg_share = get_package_share_directory('mppi_torch')
        full_model_path = os.path.join(pkg_share, model_path)

        if os.path.exists(full_model_path):
            self.dynamics_model = torch.jit.load(full_model_path, map_location=self.device)
            self.dynamics_model.eval()
            self.get_logger().info(f'Loaded model from {full_model_path}')
        else:
            self.get_logger().error(f'Model file not found at {full_model_path}')
            raise FileNotFoundError(f'Model file not found at {full_model_path}')

        self.env = RobotDynamicsEnv(self.dynamics_model, self.device)

        self.planner = MPPIPlanner(
            env=self.env,
            planning_horizon=planning_horizon,
            optim_steps=optim_steps,
            num_candidates=num_candidates,
            top_k=top_k
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

        self.get_logger().info('MPPI Node initialized with torchrl.MPPIPlanner')

    def path_callback(self, msg):
        self.current_path = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            self.current_path.append(torch.tensor([x, y], device=self.device))

        self.env.set_reference_path(self.current_path)

    def obstacle_callback(self, msg):
        self.obstacles = []
        data = np.array(msg.data)
        num_obstacles = len(data) // 3
        for i in range(num_obstacles):
            x = data[i * 3]
            y = data[i * 3 + 1]
            self.obstacles.append(torch.tensor([x, y], device=self.device))

        self.env.set_obstacles(self.obstacles)

    def control_loop(self):
        if len(self.current_path) == 0:
            return

        state_td = TensorDict({
            "observation": self.current_state,
        }, batch_size=[])

        with torch.no_grad():
            action_td = self.planner.planning(state_td)

        action = action_td["action"]

        cmd = Twist()
        cmd.linear.x = float(action[0])
        cmd.angular.z = float(action[1])
        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = MPPINode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
