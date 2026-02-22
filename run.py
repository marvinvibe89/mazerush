import os
import datetime
import statistics
import collections
import yaml
import argparse
import pygame

from agent_utils import Agent, DeepQAgent, RandomAgent, HumanAgent, NothingAgent, ActionStep
from mazerush_env import MazerushEnv


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------

def _build_agents(
    player_configs: list[dict],
    action_space,
    num_states: list[int],
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
        elif ptype == "NothingAgent":
            agents.append(NothingAgent(action_space))
        elif ptype == "DeepQAgent":
            agents.append(DeepQAgent(
                action_space,
                num_states=num_states,
                **agent_config,
            ))
        else:
            raise ValueError(f"Unknown player type: {ptype}")
    return agents


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    agents: list[Agent],
    env: MazerushEnv,
    train: bool = False,
) -> tuple[list[list[ActionStep]], list[float], list[bool]]:
    """Run a single Mazerush episode.

    Returns per-player action step lists, final cumulative rewards, and win status.
    """
    obs_n, info = env.reset()
    per_player_steps: list[list[ActionStep]] = [[] for _ in agents]
    cumulative_rewards = [0.0] * len(agents)
    is_wins = [False] * len(agents)

    while True:
        # Collect events for human agents
        events = env.get_key_events()
        for agent in agents:
            if isinstance(agent, HumanAgent):
                for event in events:
                    agent.key_listener(event)

        # Check for quit
        for event in events:
            if event.type == pygame.QUIT:
                env.close()
                return per_player_steps, cumulative_rewards, is_wins

        # Select actions
        action_n = [
            agent.select_action(obs, train)
            for agent, obs in zip(agents, obs_n)
        ]
        if not all(0 <= a < env.action_space.n for a in action_n):
            raise ValueError("Invalid action selected")

        prev_obs_n = obs_n
        obs_n, reward_n, done_n, truncated_n, info_n = env.step(action_n)

        for i in range(len(agents)):
            done = done_n[i] or truncated_n[i]
            per_player_steps[i].append(
                ActionStep(action_n[i], prev_obs_n[i], obs_n[i], reward_n[i], done)
            )
            cumulative_rewards[i] += reward_n[i]
            if info_n[i].get("result") == "win":
                is_wins[i] = True

        if any(done_n) or any(truncated_n):
            return per_player_steps, cumulative_rewards, is_wins


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    agents: list[Agent],
    env: MazerushEnv,
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
    recent_rewards: list[collections.deque] = [
        collections.deque(maxlen=200) for _ in agents
    ]

    for ep in range(num_episodes):
        per_player_steps, cumulative_rewards, is_wins = run_episode(
            agents, env, train=True,
        )

        # Register steps & train DeepQAgents
        for i, agent in enumerate(agents):
            if isinstance(agent, DeepQAgent):
                agent.register_action_steps(per_player_steps[i])
                won = is_wins[i]
                recent_wins[i].append(1 if won else 0)
                recent_rewards[i].append(cumulative_rewards[i])
                if won:
                    win_counts[i] += 1

        if (ep + 1) % train_frequency == 0:
            for i, agent in enumerate(agents):
                if isinstance(agent, DeepQAgent):
                    loss_hist = agent.train()
                    wr = sum(recent_wins[i]) / max(len(recent_wins[i]), 1) * 100
                    avg_reward = sum(recent_rewards[i]) / max(len(recent_rewards[i]), 1)
                    avg_loss = statistics.mean(loss_hist) if loss_hist else 0.0
                    print(
                        f"[Player {i}] Ep {ep} - Loss: {avg_loss:.6f} "
                        f"- Win Rate: {wr:.1f}% - Avg Reward: {avg_reward:.2f} "
                        f"- Total Wins: {win_counts[i]} "
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

def main():
    """Main function to run the Mazerush game."""

    parser = argparse.ArgumentParser(description="Mazerush")
    parser.add_argument("--config", type=str, default="config/mazerush.yaml", help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Mode: train or test")
    parser.add_argument("--episodes-override", type=int, default=None, help="Override num_episodes for quick testing")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    env_config = config["env"].get("config") or {}
    player_configs = config.get("players", [{"type": "RandomAgent"}, {"type": "RandomAgent"}])
    num_players = len(player_configs)

    render_mode = "human" if args.mode == "test" else None
    env = MazerushEnv(num_players=num_players, render_mode=render_mode, **env_config)

    num_states = env.observation_space.nvec.tolist()
    agent_config = config.get("agent", {})
    agents = _build_agents(player_configs, env.action_space, num_states, agent_config)

    # Resume DeepQAgents
    resume_paths = args.resume or config.get("training", {}).get("resume_paths")
    if resume_paths:
        resume_paths = resume_paths.split(",")
        resume_idx = 0
        for i, agent in enumerate(agents):
            if isinstance(agent, DeepQAgent):
                resume_path = resume_paths[resume_idx]
                resume_idx = (resume_idx + 1) % len(resume_paths)
                if os.path.exists(resume_path):
                    agent.load(resume_path)
                    print(f"Resumed player {i} from {resume_path}")
                else:
                    raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    if args.mode == "train":
        training_cfg = config.get("training", {})
        num_episodes = args.episodes_override or training_cfg.get("num_episodes", 1000)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = f"out/{timestamp}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to {checkpoint_dir}")

        train(
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
            per_player_steps, rewards, is_wins = run_episode(agents, env, train=False)
            print(f"Episode finished. Rewards: {rewards}, Wins: {is_wins}")
            if not pygame.display.get_init():
                break

    env.close()

if __name__ == "__main__":
    main()
