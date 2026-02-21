import os
import datetime
import gymnasium as gym
import statistics
import time
import collections
import yaml
import argparse
from typing import Callable

from agent_utils import Agent, DeepQAgent, ActionStep


def run_episode(
        agent: Agent,
        env: gym.Env,
        shape_reward: Callable[[float, bool, bool, int, set[int]], float] | None = None,
        wait_time: float = 0.1,
        train: bool = False,
        verbose: bool = False,
    ):
    state, info = env.reset()

    if verbose:
        print(state)
        print(info)

    if wait_time > 0:
        time.sleep(wait_time)

    state_history = set()
    prev_state = state
    action_steps = []

    while True:
        action = agent.select_action(state, train)
        state, reward, terminated, truncated, info = env.step(action)
        if shape_reward is not None:
            reward = shape_reward(reward, truncated, terminated, state, state_history)
        done = terminated or truncated
        action_steps.append(ActionStep(action, prev_state, state, reward, done))
        prev_state = state
        state_history.add(state)

        if verbose:
            print(' ============= ')
            print(f'{state=}')
            print(f'{reward=}')
            print(f'{terminated=}')
            print(f'{truncated=}')
            print(f'{info=}')
        if wait_time > 0:
            time.sleep(wait_time)

        if terminated or truncated:
            return action_steps, reward
    return None, None
    

def train(
        agent: DeepQAgent,
        env: gym.Env,
        wait_time: float,
        num_episodes: int,
        shape_reward: Callable[[float, bool, bool, int, set[int]], float] | None = None,
        checkpoint_dir: str | None = None,
        save_interval: int = 1000,
        train_frequency: int = 100,
    ):
    win_history = collections.deque(maxlen=1000)
    total_wins = 0

    for it in range(num_episodes):
        action_steps, final_reward = run_episode(agent, env, shape_reward, wait_time, train=True)
        agent.register_action_steps(action_steps)
        
        is_win = final_reward > 0
        win_history.append(1 if is_win else 0)
        if is_win:
            total_wins += 1

        if (it + 1) % train_frequency == 0:
            loss_history = agent.train()
            win_rate = sum(win_history) / len(win_history) * 100 if len(win_history) > 0 else 0.0
            print(f'Episode {it} - Loss: {statistics.mean(loss_history):.6f} - Win Rate: {win_rate:.1f}% ({sum(win_history)} wins) - Total Wins: {total_wins} - Epsilon: {agent._epsilon:.4f}')

        if checkpoint_dir and (it + 1) % save_interval == 0:
            path = os.path.join(checkpoint_dir, f'checkpoint_{it + 1}.pt')
            agent.save(path)
            print(f'Saved checkpoint to {path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Q-Learning")
    parser.add_argument("--config", type=str, default="config/taxi.yaml", help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Mode: train or test")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    map_name = config["env"]["name"]
    map_config = config["env"]["config"] or {}
    
    shape_reward = None

    # Initialize environment
    render_mode = "human" if args.mode == "test" else None
    env = gym.make(map_name, **map_config, render_mode=render_mode)
    
    # Initialize Agent with config values
    agent_config = config["agent"]
    deep_q_agent = DeepQAgent(
        env.action_space, 
        env.observation_space.n, 
        **agent_config,
    )
    
    # Resume capability
    resume_path = args.resume or config["training"].get("resume_path")
    if resume_path and os.path.exists(resume_path):
        deep_q_agent.load(resume_path)
        print(f"Resumed from {resume_path}")
    elif args.mode == "test" and not resume_path:
        print("Warning: Running in test mode without a checkpoint/resume path.")

    if args.mode == "train":
        # Checkpoint setup
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = f"out/{timestamp}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to {checkpoint_dir}")

        train(
            deep_q_agent, 
            env, 
            **config["training"],
            checkpoint_dir=checkpoint_dir, 
        )
    
    elif args.mode == "test":
        for _ in range(100):
            run_episode(deep_q_agent, env, shape_reward, wait_time=0.1, train=False, verbose=True)
