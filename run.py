import os
import datetime
import gymnasium as gym
import statistics
import time
import collections
import yaml
import argparse
from typing import Callable

from agent_utils import Agent, DeepQAgent, RandomAgent, HumanAgent, ActionStep


# ---------------------------------------------------------------------------
# Single-agent helpers (Taxi etc.)
# ---------------------------------------------------------------------------

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


def train_single(
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


# ---------------------------------------------------------------------------
# Multi-agent helpers (Mazerush)
# ---------------------------------------------------------------------------

def _build_agents(
    player_configs: list[dict],
    action_space,
    obs_dim: int,
    agent_config: dict,
) -> list[Agent]:
    """Instantiate agents from the players list in the config."""
    agents: list[Agent] = []
    for pc in player_configs:
        ptype = pc["type"]
        if ptype == "HumanAgent":
            agents.append(HumanAgent(action_space))
        elif ptype == "RandomAgent":
            agents.append(RandomAgent(action_space))
        elif ptype == "DeepQAgent":
            agents.append(DeepQAgent(
                action_space,
                state_dim=obs_dim,
                **agent_config,
            ))
        else:
            raise ValueError(f"Unknown player type: {ptype}")
    return agents


def run_mazerush_episode(
    agents: list[Agent],
    env,
    train: bool = False,
) -> tuple[list[list[ActionStep]], list[float]]:
    """Run a single Mazerush episode.

    Returns per-player action step lists and final rewards.
    """
    obs_n, info = env.reset()
    per_player_steps: list[list[ActionStep]] = [[] for _ in agents]
    cumulative_rewards = [0.0] * len(agents)

    while True:
        # Collect events for human agents
        events = env.get_key_events()
        for agent in agents:
            if isinstance(agent, HumanAgent):
                for event in events:
                    agent.key_listener(event)

        # Check for quit
        import pygame
        for event in events:
            if event.type == pygame.QUIT:
                env.close()
                return per_player_steps, cumulative_rewards

        # Select actions
        action_n = [
            agent.select_action(obs, train)
            for agent, obs in zip(agents, obs_n)
        ]
        # Replace invalid actions (e.g. HumanAgent no-op = -1) with a
        # value outside the movement/shoot range so the env does nothing.
        action_n = [a if 0 <= a < env.action_space.n else -1 for a in action_n]

        prev_obs_n = obs_n
        obs_n, reward_n, done_n, truncated_n, info_n = env.step(action_n)

        for i in range(len(agents)):
            done = done_n[i] or truncated_n[i]
            per_player_steps[i].append(
                ActionStep(action_n[i], prev_obs_n[i], obs_n[i], reward_n[i], done)
            )
            cumulative_rewards[i] += reward_n[i]

        if any(done_n) or any(truncated_n):
            return per_player_steps, cumulative_rewards


def train_mazerush(
    agents: list[Agent],
    env,
    num_episodes: int,
    checkpoint_dir: str | None = None,
    save_interval: int = 1000,
    train_frequency: int = 50,
):
    """Training loop for Mazerush with multiple agents."""
    win_counts = [0] * len(agents)
    recent_wins: list[collections.deque] = [
        collections.deque(maxlen=200) for _ in agents
    ]

    for ep in range(num_episodes):
        per_player_steps, cumulative_rewards = run_mazerush_episode(
            agents, env, train=True,
        )

        # Register steps & train DeepQAgents
        for i, agent in enumerate(agents):
            if isinstance(agent, DeepQAgent):
                agent.register_action_steps(per_player_steps[i])
                won = cumulative_rewards[i] > 0
                recent_wins[i].append(1 if won else 0)
                if won:
                    win_counts[i] += 1

        if (ep + 1) % train_frequency == 0:
            for i, agent in enumerate(agents):
                if isinstance(agent, DeepQAgent):
                    loss_hist = agent.train()
                    wr = sum(recent_wins[i]) / max(len(recent_wins[i]), 1) * 100
                    avg_loss = statistics.mean(loss_hist) if loss_hist else 0.0
                    print(
                        f"[Player {i}] Ep {ep} - Loss: {avg_loss:.6f} "
                        f"- Win Rate: {wr:.1f}% - Total Wins: {win_counts[i]} "
                        f"- Eps: {agent._epsilon:.4f}"
                    )

        if checkpoint_dir and (ep + 1) % save_interval == 0:
            for i, agent in enumerate(agents):
                if isinstance(agent, DeepQAgent):
                    path = os.path.join(checkpoint_dir, f'player{i}_ckpt_{ep + 1}.pt')
                    agent.save(path)
                    print(f'Saved player {i} checkpoint to {path}')

    print(f"\nTraining complete. Win counts: {win_counts}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Q-Learning / Mazerush")
    parser.add_argument("--config", type=str, default="config/mazerush.yaml", help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Mode: train or test")
    parser.add_argument("--episodes-override", type=int, default=None, help="Override num_episodes for quick testing")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    env_name = config["env"]["name"]
    env_config = config["env"].get("config") or {}

    is_mazerush = env_name == "Mazerush"

    if is_mazerush:
        # ----- Multi-agent Mazerush path -----
        from mazerush_env import MazerushEnv

        player_configs = config.get("players", [{"type": "RandomAgent"}, {"type": "RandomAgent"}])
        num_players = len(player_configs)

        render_mode = "human" if args.mode == "test" else None
        env = MazerushEnv(num_players=num_players, render_mode=render_mode, **env_config)

        obs_dim = env.observation_space.shape[0]
        agent_config = config.get("agent", {})
        agents = _build_agents(player_configs, env.action_space, obs_dim, agent_config)

        # Resume DeepQAgents
        resume_path = args.resume or config.get("training", {}).get("resume_path")
        if resume_path:
            for i, agent in enumerate(agents):
                if isinstance(agent, DeepQAgent):
                    p = resume_path.replace(".pt", f"_player{i}.pt")
                    if os.path.exists(p):
                        agent.load(p)
                        print(f"Resumed player {i} from {p}")
                    elif os.path.exists(resume_path):
                        agent.load(resume_path)
                        print(f"Resumed player {i} from {resume_path}")

        if args.mode == "train":
            training_cfg = config.get("training", {})
            num_episodes = args.episodes_override or training_cfg.get("num_episodes", 1000)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = f"out/{timestamp}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Checkpoints will be saved to {checkpoint_dir}")

            train_mazerush(
                agents, env,
                num_episodes=num_episodes,
                checkpoint_dir=checkpoint_dir,
                save_interval=training_cfg.get("save_interval", 1000),
                train_frequency=training_cfg.get("train_frequency", 50),
            )
        else:
            # Test / play mode
            print("Starting Mazerush in play mode. Close the window to exit.")
            while True:
                per_player_steps, rewards = run_mazerush_episode(agents, env, train=False)
                print(f"Episode finished. Rewards: {rewards}")
                # Re-check if window was closed
                import pygame
                if not pygame.display.get_init():
                    break
        env.close()

    else:
        # ----- Single-agent path (Taxi etc.) -----
        render_mode = "human" if args.mode == "test" else None
        env = gym.make(env_name, **(env_config or {}), render_mode=render_mode)

        agent_config = config["agent"]
        deep_q_agent = DeepQAgent(
            env.action_space,
            env.observation_space.n,
            **agent_config,
        )

        resume_path = args.resume or config["training"].get("resume_path")
        if resume_path and os.path.exists(resume_path):
            deep_q_agent.load(resume_path)
            print(f"Resumed from {resume_path}")
        elif args.mode == "test" and not resume_path:
            print("Warning: Running in test mode without a checkpoint/resume path.")

        if args.mode == "train":
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = f"out/{timestamp}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Checkpoints will be saved to {checkpoint_dir}")

            num_episodes = args.episodes_override or config["training"].get("num_episodes", 1000)
            train_single(
                deep_q_agent,
                env,
                wait_time=config["training"].get("wait_time", 0),
                num_episodes=num_episodes,
                checkpoint_dir=checkpoint_dir,
                save_interval=config["training"].get("save_interval", 1000),
                train_frequency=config["training"].get("train_frequency", 100),
            )
        elif args.mode == "test":
            for _ in range(100):
                run_episode(deep_q_agent, env, wait_time=0.1, train=False, verbose=True)
