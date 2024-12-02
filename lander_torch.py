# Heart of the project, implementation of PPO
# Date: 1th of December 2024
# Author: Ondrej Galeta 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
    parser.add_argument('--output_model_actor', type=str, default='actor_model.pth', 
                        help="Path to save the trained actor model. Default is 'actor_model.pth'.")
    parser.add_argument('--output_model_critic', type=str, default='critic_model.pth', 
                        help="Path to save the trained critic model. Default is 'critic_model.pth'.")
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
    torch.save(agent.actor.state_dict(), args.output_model_actor)
    torch.save(agent.critic.state_dict(), args.output_model_critic)
    sys.exit(0)  # Exit the program cleanly

args = parse()
# Set up signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, lambda sig, frame: save_models_on_interrupt(agent))

# Environment setup
env = gym.make('LunarLander-v2', gravity = -10)
env = gym.wrappers.TimeLimit(env)
state_dim = env.observation_space.shape[0]  # 8
action_dim = env.action_space.n  # 4 discrete actions

# Actor network
class Actor(nn.Module):
    '''
    Actor network
    input - state
    output - probabilities coresponding to different actions
    '''
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = torch.where(torch.isfinite(state), state, torch.zeros_like(state))
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.clamp(x, min=0, max=1e10) 
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.clamp(x, min=0, max=1e10) 
        x = self.fc3(x)
        x = torch.clamp(x, min=-1e10, max=1e10)       
        # Apply numerically stable softmax
        return torch.nn.functional.softmax(x, dim=-1)

# Critic network
class Critic(nn.Module):
    '''
    Critic critic network
    input - state
    output - reward aproximation for given state
    '''
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=3e-4, clip_ratio=0.2, lambd=0.95):
        '''
        lr - learning rate
        gamma - discount factor
        lambd - controls the trade-off between bias (advantage shifted) and variance (advantage noisy) in the computation of the advantage function
        '''
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.lambd = lambd

        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def get_action(self, state):
        '''
        Sample action based on actor network 
        input - state
        output - chosen action + logaritm of its probability
        '''
        # get probability distribution for action with given state
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        prob = self.actor(state).detach().numpy()[0]

        # Sample action with given probabilities
        action = np.random.choice(len(prob), p=prob)
        log_prob = np.log(prob[action])
        return action, log_prob

    def compute_advantage(self, rewards, dones, values):
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
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lambd * (1 - dones[i]) * gae
            advantages[i] = gae
        return advantages

    def update(self, states, actions, advantages, returns, old_log_probs):
        '''
        Updates actor and critic networks
        '''
        for _ in range(10):  # PPO typically uses multiple updates per batch
            # Update actor
            prob = self.actor(states)
            selected_log_probs = torch.log(prob.gather(1, actions.unsqueeze(1)).squeeze(1)) # log(pi(a|s))
            ratio = torch.exp(selected_log_probs - old_log_probs) # pi_new / pi_old
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            actor_loss = -(torch.min(ratio * advantages, clipped_ratio * advantages)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update critic
            value_preds = self.critic(states).squeeze(1)
            critic_loss = nn.MSELoss()(value_preds, returns)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

# Training Loop
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = PPOAgent(state_dim, action_dim)
laststeps = []
coef = 0

for episode in range(args.num_episodes):
    state = env.reset()[0]
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
            reward,prev_shaping = rw.get_reward_v2(next_state, info, prev_shaping)
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
            print("No reward selected or unkonowm reward. Double check name of chosen function.")
            exit(1)
        step_counter += 1

        value = agent.critic(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).item()

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        old_log_probs.append(log_prob)
        values.append(value)

        state = next_state
        total_reward += reward
        total_reward_old += reward_old

    next_value = agent.critic(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).item()
    values.append(next_value)
    advantages = agent.compute_advantage(rewards, dones, values)
    returns = advantages + values[:-1]

    # Prepare data for learning
    states_array = np.array(states) 
    actions_array = np.array(actions)
    advantages_array = np.array(advantages)
    returns_array = np.array(returns)

    states_tensor = torch.tensor(states_array, dtype=torch.float32)
    actions_tensor = torch.tensor(actions_array, dtype=torch.int64)
    advantages_tensor = torch.tensor(advantages_array, dtype=torch.float32)
    returns_tensor = torch.tensor(returns_array, dtype=torch.float32)
    old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32)

    # Update weights
    agent.update(states_tensor, actions_tensor, advantages_tensor, returns_tensor, old_log_probs_tensor)

    laststeps.append(step_counter)

    if episode + 1 > 20:
        coef = 0.00000001 * np.mean(laststeps[-20:])**3

    print(f"Episode {episode + 1}: Total Reward = {total_reward}, coef = {coef}, Steps = {step_counter}")
    with open(args.csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([episode + 1, total_reward, total_reward_old, step_counter])

# Save models
torch.save(agent.actor.state_dict(), args.output_model_actor)
torch.save(agent.critic.state_dict(), args.output_model_critic)
