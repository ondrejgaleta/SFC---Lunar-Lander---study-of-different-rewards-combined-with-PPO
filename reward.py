# Reward function definitions
# Date: 1th of December 2024
# Author: Ondrej Galeta 

import numpy as np

def get_reward_old(state, info, prev_shaping):

    reward = 0
    dist = np.sqrt(state[0] * state[0] + state[1] * state[1])
    vel = np.sqrt(state[2] * state[2] + state[3] * state[3])
    angle = abs(state[4])
    shaping = (
        - 100 * dist
        - 100 * vel
        - 100 * angle
        + 10 * state[6]
        + 10 * state[7]
    ) 
    if prev_shaping is not None:
        reward = shaping - prev_shaping
    prev_shaping = shaping

    fuel = info["m_power"] * 0.30
    fuel += info["s_power"] * 0.03

    reward -= fuel


    if info["crashed"]:
        reward = -100
    if not info["awake"]:
        reward = +100

    return reward, prev_shaping

def get_reward_v1(state, info, prev_shaping):
    # new 

    reward = 0
    dist = np.sqrt(state[0] * state[0] + state[1] * state[1])
    vel = np.sqrt(state[2] * state[2] + state[3] * state[3])
    angle = abs(state[4])
    shaping = (
        - 100 * dist
        - 100 * vel
        - 100 * angle
        + 10 * state[6]
        + 10 * state[7]
    ) 
    if prev_shaping is not None:
        reward = shaping - prev_shaping
    prev_shaping = shaping

    fuel = info["m_power"] * 0.30
    fuel += info["s_power"] * 0.03

    reward -= fuel

    if info["crashed"]:
        reward = -30*(1+vel)
        reward -= 30*(1+angle)
        reward -= 30*(1+dist)
        # reward = -100*(1+vel+angle+dist)
    if not info["awake"]:
        reward = +300
    return reward, prev_shaping


def get_reward_v2(state, info, prev_shaping):
    # increased fuel cost
    reward = 0
    dist = np.sqrt(state[0] * state[0] + state[1] * state[1])
    vel = np.sqrt(state[2] * state[2] + state[3] * state[3])
    angle = abs(state[4])
    shaping = (
        - 100 * dist
        - 100 * vel
        - 100 * angle
        + 10 * state[6]
        + 10 * state[7]
    ) 
    if prev_shaping is not None:
        reward = shaping - prev_shaping
    prev_shaping = shaping

    fuel = info["m_power"] * 0.30
    fuel += info["s_power"] * 0.03

    reward -= 5*fuel

    if info["crashed"]:
        # reward = -100*(1+vel+angle+dist)
        reward = -30*(1+vel)
        reward -= 30*(1+angle)
        reward -= 30*(1+dist)
        # reward = -100
    if not info["awake"]:
        reward = +300

    return reward, prev_shaping

def get_reward_v21(state, info, prev_shaping):
    # increased fuel cost
    reward = 0
    dist = np.sqrt(state[0] * state[0] + state[1] * state[1])
    vel = np.sqrt(state[2] * state[2] + state[3] * state[3])
    angle = abs(state[4])
    shaping = (
        - 100 * dist
        - 100 * vel
        - 100 * angle
        + 10 * state[6]
        + 10 * state[7]
    ) 
    if prev_shaping is not None:
        reward = shaping - prev_shaping
    prev_shaping = shaping

    fuel = info["m_power"] * 0.30
    fuel += info["s_power"] * 0.03

    reward -= 5*fuel

    if info["crashed"]:
        reward = -100
    if not info["awake"]:
        reward = +100

    return reward, prev_shaping


def get_reward_v3(state, info, episode, prev_shaping): #episode
    reward = 0
    dist = np.sqrt(state[0] * state[0] + state[1] * state[1])
    vel = np.sqrt(state[2] * state[2] + state[3] * state[3])
    angle = abs(state[4])
    shaping = (
        - 100 * dist
        - 100 * vel
        - 100 * angle
        + 10 * state[6]
        + 10 * state[7]
    ) 
    # reward = shaping
    if prev_shaping is not None:
        reward = shaping - prev_shaping
    prev_shaping = shaping

    fuel = info["m_power"] * 0.30
    fuel += info["s_power"] * 0.03

    reward -= fuel*(1+episode/150)

    if info["crashed"]:
        # reward = -100*(1+vel+angle+dist)
        reward = -30*(1+vel)
        reward -= 30*(1+angle)
        reward -= 30*(1+dist)
        reward /= (1+(episode/150))
        if reward > -150: 
            reward = -150
    if not info["awake"]:
        reward = +500*(1+episode/150)

    return reward, prev_shaping

def get_reward_v4(state, info, coef, prev_shaping): 
    reward = 0
    dist = np.sqrt(state[0] * state[0] + state[1] * state[1])
    vel = np.sqrt(state[2] * state[2] + state[3] * state[3])
    angle = abs(state[4])
    shaping = (
        - 100 * dist
        - 100 * vel 
        - 100 * angle
        + 10 * state[6]
        + 10 * state[7]
    ) 
    if prev_shaping is not None:
        reward = shaping - prev_shaping
    prev_shaping = shaping

    fuel = info["m_power"] * 0.30
    fuel += info["s_power"] * 0.03

    reward -= fuel*coef

    if info["crashed"]:
        reward = -100
    if not info["awake"]:
        reward = +1000*(1+coef)
    return reward, prev_shaping

def get_reward_v5(state, info, coef, prev_shaping): 
    reward = 0
    dist = np.sqrt(state[0] * state[0] + state[1] * state[1])
    vel = np.sqrt(state[2] * state[2] + state[3] * state[3])
    angle = abs(state[4])
    shaping = (
        - 100 * dist
        - 100 * vel 
        - 100 * angle
        + 10 * state[6]
        + 10 * state[7]
    ) 
    # reward = shaping
    if prev_shaping is not None:
        reward = shaping - prev_shaping
    prev_shaping = shaping

    fuel = info["m_power"] * 0.30
    fuel += info["s_power"] * 0.03

    reward -= fuel*coef

    if info["crashed"]:
        reward = -30*(1+vel)
        reward -= 30*(1+angle)
        reward -= 30*(1+dist)
        # reward = -100*(1+vel+angle+dist)
        reward /= (coef+0.5)
        if reward > -150:
            reward = -150
    if not info["awake"]:
        reward = +1000*(1+coef)

    return reward, prev_shaping

def get_reward_v6(state, info, coef, prev_shaping): 
    reward = 0
    dist = np.sqrt(state[0] * state[0] + state[1] * state[1])
    vel = np.sqrt(state[2] * state[2] + state[3] * state[3])
    angle = abs(state[4])
    shaping = (
        - 100 * dist
        - 100 * vel 
        - 100 * angle
        + 10 * state[6]
        + 10 * state[7]
    ) 
    if prev_shaping is not None:
        reward = shaping - prev_shaping
    prev_shaping = shaping

    fuel = info["m_power"] * 0.30
    fuel += info["s_power"] * 0.03

    reward -= fuel*coef

    if info["crashed"]:
        reward = -80*(1+vel)
        reward /= (coef+0.5)
        if reward > -150:
            reward = -150
    if not info["awake"]:
        reward = +1000*(1+coef)
    return reward, prev_shaping
