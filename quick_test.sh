#!/bin/bash

sudo python3 lander_torch.py --reward_model improved_crash_landing_increased_fuel_price_relative --num_episodes 1000 --output_model_actor test_actor.pth 

sudo python3 show_torch.py --actor_model test_actor.pth --video_name test