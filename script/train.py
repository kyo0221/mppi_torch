#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from network import Network


class RobotDynamicsDataset(Dataset):
    def __init__(self, data_file):
        data = np.load(data_file)
        self.states = torch.FloatTensor(data['states'])
        self.actions = torch.FloatTensor(data['actions'])
        self.next_states = torch.FloatTensor(data['next_states'])

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        next_state = self.next_states[idx]
        state_delta = next_state - state

        model_input = torch.cat([state, action])
        return model_input, state_delta


def generate_synthetic_data(num_samples=10000, dt=0.1):
    states = []
    actions = []
    next_states = []

    for _ in range(num_samples):
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-10, 10)
        theta = np.random.uniform(-np.pi, np.pi)

        v = np.random.uniform(0, 2.0)
        delta = np.random.uniform(-0.5, 0.5)

        dx = v * np.cos(theta) * dt
        dy = v * np.sin(theta) * dt
        dtheta = delta * dt

        next_x = x + dx
        next_y = y + dy
        next_theta = theta + dtheta

        states.append([x, y, theta])
        actions.append([v, delta])
        next_states.append([next_x, next_y, next_theta])

    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'next_states': np.array(next_states)
    }


def train_model(data_file, output_path, num_epochs=100, batch_size=64, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if not os.path.exists(data_file):
        print(f'Data file not found. Generating synthetic data...')
        data = generate_synthetic_data()
        np.savez(data_file, **data)
        print(f'Synthetic data saved to {data_file}')

    dataset = RobotDynamicsDataset(data_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Network().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('Starting training...')
    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')

    model.eval()
    model.cpu()
    scripted_model = torch.jit.script(model)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scripted_model.save(output_path)
    print(f'Model saved to {output_path}')


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.dirname(script_dir)

    data_file = os.path.join(pkg_dir, 'data', 'robot_dynamics_data.npz')
    output_path = os.path.join(pkg_dir, 'weights', 'dynamics_model.pt')

    os.makedirs(os.path.join(pkg_dir, 'data'), exist_ok=True)

    train_model(data_file, output_path)
