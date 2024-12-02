# Heart of the project, implementation of PPO
# Date: 1th of December 2024
# Author: Ondrej Galeta 

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import reward as rw
import signal
import csv
import argparse
import sys
sys.path.insert(0,'./gym')
import gym



def parse():
    '''
    Function sets valid arguments and reads them
    '''
    parser = argparse.ArgumentParser(
        description="Process training parameters for a reinforcement learning model.",
        epilog="Example usage: python train_model.py --csv_file data.csv --num_episodes 500")
    parser.add_argument('--csv_file', type=str, default='data.csv', 
                        help="Path to the output CSV file containing collected data during training. Default is 'data.csv'.")
    parser.add_argument('--output_model_actor', type=str, default='actor_model.h5', 
                        help="Path to save the trained actor model. Default is 'actor_model.h5'.")
    parser.add_argument('--output_model_critic', type=str, default='critic_model.h5', 
                        help="Path to save the trained critic model. Default is 'critic_model.h5'.")
    parser.add_argument('--max_number_steps', type=int, default=1000, 
                        help="Maximum number of steps allowed per episode during training. Default is 1000.")
    parser.add_argument('--num_episodes', type=int, default=1000, 
                        help="Number of training episodes to run. Default is 1000.")
    parser.add_argument('--reward_model', type=str, default='baseline', 
                        help="Type of reward model to use. Default is 'baseline'.")
    parser.add_argument('--gamma', type=float, default=0.99, 
                        help="Discount factor for future rewards. Default is 0.99.")
    parser.add_argument('--lr', type=float, default=0.0003, 
                        help="Learning rate for the optimizer. Default is 0.0003.")
    parser.add_argument('--clip_ratio', type=float, default=0.2, 
                        help="Clipping ratio for PPO. Default is 0.2.")
    parser.add_argument('--lambd', type=float, default=0.95, 
                        help="Lambda for GAE (Generalized Advantage Estimation). Default is 0.95.")
    return parser.parse_args()

def save_models_on_interrupt(agent):
    '''
    Save models when interrupt signal occurs 
    '''
    agent.actor.save(args.output_model_actor)
    agent.critic.save(args.output_model_critic)
    sys.exit(0)  # Exit the program cleanly

args = parse()
# Set up signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, lambda sig, frame: save_models_on_interrupt(agent))

# Environment setup
env = gym.make('LunarLander-v2', gravity = -10)
env = gym.wrappers.TimeLimit(env)
state_dim = env.observation_space.shape[0]  # 8
action_dim = env.action_space.n  # 4 discrete actions

def build_actor(state_dim, action_dim):
    '''
    Actor network
    input - state
    output - probabilities coresponding to different actions
    '''
    inputs = layers.Input(shape=(state_dim,))
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(action_dim)(x)
    prob = tf.nn.softmax(x)
    return tf.keras.Model(inputs, prob)

def build_critic(state_dim):
    '''
    Critic critic network
    input - state
    output - reward aproximation for given state
    '''
    inputs = layers.Input(shape=(state_dim,))
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(64, activation="relu")(x)
    value = layers.Dense(1)(x)
    return tf.keras.Model(inputs, value)

# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, gamma=args.gamma, lr=args.lr, clip_ratio=args.clip_ratio, lambd=args.lambd): #lr = 0.0003
        '''
        lr - learning rate
        gamma - discount factor
        lambd - controls the trade-off between bias (advantage shifted) and variance (advantage noisy) in the computation of the advantage function (TODO controls stability of learning?)
        '''
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.lambd = lambd
        self.actor = build_actor(state_dim, action_dim)
        self.critic = build_critic(state_dim)
        self.actor_optimizer = Adam(learning_rate=lr)
        self.critic_optimizer = Adam(learning_rate=lr)

    def get_action(self, state: np.ndarray) -> [int, float]:
        '''
        Sample action based on actor network 
        input - state
        output - chosen action + logaritm of its probability
        '''
        # get probability distribution for action with given state
        state = np.expand_dims(state, axis=0).astype(np.float32)
        prob = self.actor(state)

        # Sample action with given probabilities
        action = np.random.choice(action_dim, p=prob.numpy()[0])

        # return action with its probability
        return action, tf.math.log(prob[0, action])

    def compute_advantage(self, rewards: np.ndarray, dones: np.ndarray, values: np.ndarray) -> np.ndarray:
        '''
        Generalized Advantage Estimation
        rewards - array of gained rewards
        dones - array of flags, flag is one if episode ended at the appropriate step
        values - aproximated values by critic
        note that same indexes of all imput arrays correspond to same time step 
        '''
        advantages = np.zeros_like(rewards)
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i+1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lambd * (1 - dones[i]) * gae
            advantages[i] = gae
        return advantages

    def update(self, states, actions, advantages, returns, old_log_probs):
        '''
        Updates actor and critic networks
        '''
        for _ in range(10):  # PPO typically uses multiple updates per batch

            # Update agent
            with tf.GradientTape() as tape:
                prob = self.actor(states)
                selected_log_probs = tf.reduce_sum(tf.math.log(prob) * tf.one_hot(actions, action_dim), axis=1)
                ratio = tf.exp(selected_log_probs - old_log_probs)
                clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            # Update critic
            with tf.GradientTape() as tape:
                critic_loss = tf.reduce_mean((returns - self.critic(states)) ** 2) # MSE
            critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

# Training loop
agent = PPOAgent(state_dim, action_dim)
last10rewards = []
laststeps = []
coef = 0

for episode in range(args.num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    total_reward_old = 0
    states, actions, rewards, dones, old_log_probs, values = [], [], [], [], [], []
    step_counter = 0
    prev_shaping = None

    # Episode cycle, get state -> take action -> get state...
    while not done and step_counter < args.max_number_steps:
        action, log_prob = agent.get_action(state)

        next_state, reward_old, done, _, info = env.step(action)
        
        if args.reward_model == "baseline":
            reward = reward_old
        elif  args.reward_model == "improved_crash_lading":
            reward, prev_shaping = rw.get_reward_v1(next_state, info, prev_shaping)
        elif  args.reward_model == "improved_crash_lading_increased_fuel_price":
            reward, prev_shaping = rw.get_reward_v2(next_state, info, prev_shaping)
        elif  args.reward_model == "increased_fuel_price":
            reward, prev_shaping = rw.get_reward_v21(next_state, info, prev_shaping)
        elif  args.reward_model == "improved_crash_landing_increased_fuel_price_episode":
            reward, prev_shaping = rw.get_reward_v3(next_state, info, episode, prev_shaping)
        elif  args.reward_model == "increased_fuel_price_relative":
            reward, prev_shaping = rw.get_reward_v4(next_state, info, coef, prev_shaping)
        elif  args.reward_model == "improved_crash_landing_increased_fuel_price_relative":
            reward, prev_shaping = rw.get_reward_v5(next_state, info, coef, prev_shaping)
        elif  args.reward_model == "improved_crash_landing_increased_fuel_price_relative_v2":
            reward, prev_shaping = rw.get_reward_v6(next_state, info, coef, prev_shaping)
        else:
            print("No reward selected.")
        step_counter += 1

        value = agent.critic(np.expand_dims(state, axis=0)).numpy()[0, 0]

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        old_log_probs.append(log_prob)
        values.append(value)

        state = next_state

        total_reward += reward
        total_reward_old += reward_old

    next_value = agent.critic(np.expand_dims(state, axis=0)).numpy()[0, 0]
    values = values + [next_value]
    advantages = agent.compute_advantage(rewards, dones, values)
    returns = advantages + values[:-1]

    # Update weights
    agent.update(
        np.array(states),
        np.array(actions),
        np.array(advantages, dtype=np.float32),
        np.array(returns, dtype=np.float32),
        np.array(old_log_probs)
    )
    
    last10rewards.append(total_reward_old)

    # calculate coef
    laststeps.append(step_counter)
    if episode+1 > 20: #10
        coef = 0.00000001 * np.mean(laststeps[-20:])**3

    print(f"Episode {episode + 1}: Total Reward = {total_reward}, coef= {coef},Steps = {step_counter}")
    data = [episode + 1, total_reward, total_reward_old, step_counter]  # Replace with actual data

    # Append the data as a new row to the CSV file
    with open(args.csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

# Save the actor models
agent.actor.save(args.output_model_actor)
agent.critic.save(args.output_model_critic)
