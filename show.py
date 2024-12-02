# Script to create a video of a Lunar Lander to try land given pretrained model
# Date: 1th of December 2024
# Author: Ondrej Galeta 

import numpy as np
import tensorflow as tf
import argparse
import os
import sys
sys.path.insert(0,'./gym')
import gym


# Evaluate the trained agent
def run_final_episode(actor_model, env, video_tag):

    os.makedirs(args.output_path, exist_ok=True)

    state, _ = env.reset()
    done = False
    total_reward = 0
    step_counter = 0

    env = gym.wrappers.RecordVideo(env, args.output_path, episode_trigger=lambda episode_id: True, name_prefix=video_tag)
    
    while not done:
        state = np.expand_dims(state, axis=0).astype(np.float32)
        action_probs = actor_model(state)[0]  # Predict action probabilities
        
        # action = np.random.choice(len(action_probs), p=action_probs.numpy())  # Sample action
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
parser.add_argument("--actor_model", type=str, help="Path to the trained actor model. Default is actor_model.h5", default='actor_model.h5')

args = parser.parse_args()

# Create environment and trained PPO agent
env = gym.make('LunarLander-v2', render_mode="rgb_array", gravity = -10.0)

# Load actor
actor_model = tf.keras.models.load_model(args.actor_model)

# Run and visualize the final episode
run_final_episode(actor_model, env, args.video_tag)
