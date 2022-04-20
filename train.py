#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Train a model on the given environment. """

import argparse
from typing import Optional
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from unityagents import UnityEnvironment
from ddqn import Agent

EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995


def main(env: str, episodes: int = 100, seed: Optional[int] = None) -> None:
    env = UnityEnvironment(file_name=env)
    seed = seed if seed is not None else np.random.randint(0, 10000)

    writer = SummaryWriter()

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]
    state_size = len(env_info.vector_observations[0])
    action_size = brain.vector_action_space_size
    num_agents = len(env_info.agents)
    agent = Agent(state_size, action_size, seed=seed, writer=writer)

    eps = EPS_START
    ep_avg_scores = []
    solved = False
    progress = tqdm(range(1, episodes + 1), desc="Training", ncols=80)
    for i_episode in progress:
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        while True:
            actions = agent.act(states, eps)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            scores += rewards
            if np.any(dones):
                break
        avg_score = np.mean(scores)
        eps = max(EPS_END, EPS_DECAY * eps)
        writer.add_scalar("score", avg_score, i_episode)
        writer.add_scalar("epsilon", eps, i_episode)

        # if i_episode % 10 == 0:
        #     agent.save_model(f"models/checkpoint_{i_episode}.pth")

        ep_avg_scores.append(avg_score)
        progress.set_postfix(avg_score=avg_score, last_100_avg=np.mean(ep_avg_scores[-100:]))

        if not solved and np.mean(ep_avg_scores[-100:]) > 13.0:
            print(
                f"Solved in {i_episode} episodes with a last 100 episode avg score of {np.mean(ep_avg_scores[-100:])}!")
            agent.save_model("models/solved.pth")
            solved = True

    # Save final model
    print(f"Final score of last 100 episodes: {np.mean(ep_avg_scores[-100:])}")
    agent.save_model("models/final.pth")

    env.close()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN Agent')
    parser.add_argument('--env', type=str, default='env/Banana_Linux/Banana.x86_64',
                        help='Path to the environment.  Default: env/Banana_Linux/Banana.x86_64')
    parser.add_argument('--episodes', type=int, default=600, help='Number of episodes to run.  Default: 600')
    parser.add_argument('--seed', type=int, required=False, help='Random seed. Default: None (random)')

    args = parser.parse_args()
    main(args.env, args.episodes, args.seed)
