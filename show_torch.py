# Script to create a video of a Lunar Lander to try land given pretrained model
# Date: 1th of December 2024
# Author: Ondrej Galeta 

import numpy as np
import torch
import argparse
import os
import torch.nn as nn
import sys
sys.path.insert(0,'./gym')
import gym

# Define the Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.clamp(x, min=0, max=1e10) 
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.clamp(x, min=0, max=1e10) 
        x = self.fc3(x)
        x = torch.clamp(x, min=-1e10, max=1e10)       
        # Apply numerically stable softmax
        return torch.nn.functional.softmax(x, dim=-1)
    
# Evaluate the trained agent
def run_final_episode(actor_model, env, video_tag):

    os.makedirs(args.output_path, exist_ok=True)

    state, _ = env.reset()
    done = False
    total_reward = 0
    step_counter = 0

    env = gym.wrappers.RecordVideo(env, args.output_path, episode_trigger=lambda episode_id: True, name_prefix=video_tag)
    
    while not done:
        # Convert state to PyTorch tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Predict action probabilities
        with torch.no_grad():
            action_probs = actor_model(state_tensor).squeeze(0).numpy()
        
        # action = np.random.choice(len(action_probs), p=action_probs)  # Sample action
        action = np.argmax(action_probs)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Update state for the next step
        state = next_state
        total_reward += reward
        step_counter += 1

    print(f"Final Episode: Total Reward = {total_reward}, Steps = {step_counter}")

    # Close the environment rendering window
    env.close()

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run a trained PPO agent and save a video.")
parser.add_argument("--output_path", type=str, help="Directory to save the video. Default is ./video", default='./video')
parser.add_argument("--video_tag", type=str, help="Name tag of the output video. Default is lunar_lander", default='lunar_lander')
parser.add_argument("--actor_model", type=str, help="Path to the trained actor model. Default is actor_model.pth", default='actor_model.pth')

args = parser.parse_args()


# Create environment and trained PPO agent
env = gym.make('LunarLander-v2', render_mode="rgb_array", gravity = -10.0)

# Load actor weights
actor_model = Actor(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
actor_model.load_state_dict(torch.load(args.actor_model))

# Run and visualize the final episode
run_final_episode(actor_model, env, args.video_tag)
