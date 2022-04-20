#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Script to play the game with the specified model. """

import argparse
import numpy as np
import time
from unityagents import UnityEnvironment
from ddqn import Agent


def main(env: str, model_path: str, fps: int = 30) -> None:
    env = UnityEnvironment(file_name=env)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]
    state_size = len(env_info.vector_observations[0])
    action_size = brain.vector_action_space_size
    num_agents = len(env_info.agents)

    agent = Agent(state_size, action_size)
    agent.load_model(model_path)
    print(f"Loaded model from {model_path}")

    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    scores = np.zeros(num_agents)
    now = time.time()
    while True:
        actions = agent.act(states)
        env_info = env.step(actions)[brain_name]
        states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done

        scores += rewards
        if np.any(dones):
            break

        # Sleep for the remaining frame time
        elapsed = time.time() - now
        if elapsed < 1 / fps:
            sleep_time = (1 / fps) - elapsed
            time.sleep(sleep_time)
        now = time.time()

    avg_score = np.mean(scores)
    print(f"Game Over.  Final score: {np.sum(scores)}  Avg Score: {avg_score}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN Agent')
    parser.add_argument('--env', type=str, default='env/Banana_Linux/Banana.x86_64',
                        help='Path to the environment.  Default: env/Banana_Linux/Banana.x86_64')
    parser.add_argument('--model_path', type=str, default='models/final.pth',
                        help='Path to the model.  Default: models/final.pth')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second.  Default: 30')

    args = parser.parse_args()
    main(args.env, args.model_path, args.fps)
