import os
import time
import datetime
import random
import statistics
import collections
import yaml
import argparse
import pygame

from agent_utils import Agent, DeepQAgent, PPOAgent, RandomAgent, HumanAgent, NothingAgent, ActionStep
from mazerush_env import MazerushEnv


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------

def _build_agents(
    player_configs: list[dict],
    action_space,
    num_states: int,
    agent_config: dict,
    ppo_agent_config: dict | None = None,
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
        elif ptype == "PPOAgent":
            agents.append(PPOAgent(
                action_space,
                num_states=num_states,
                **(ppo_agent_config or {}),
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
) -> tuple[list[list[ActionStep]], list[float], list[str]]:
    """Run a single Mazerush episode.

    Returns per-player action step lists, final cumulative rewards, and win status.
    """
    obs_n, info = env.reset()
    for i, agent in enumerate(agents):
        agent.set_player(env.players[i])
    per_player_steps: list[list[ActionStep]] = [[] for _ in agents]
    cumulative_rewards = [0.0] * len(agents)
    results = [""] * len(agents)
    is_rendered = any(isinstance(agent, HumanAgent) for agent in agents)

    while True:
        start_time = time.time()
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
                return per_player_steps, cumulative_rewards, results

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
            results[i] = info_n[i].get("result")

        if any(done_n) or any(truncated_n):
            return per_player_steps, cumulative_rewards, results
        
        if is_rendered and time.time() - start_time < 1 / env.fps:
            time.sleep(1 / env.fps - (time.time() - start_time))



# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    agents: list[Agent],
    env: MazerushEnv,
    num_episodes: int,
    self_play: bool = False,
    checkpoint_dir: str | None = None,
    save_interval: int = 1000,
    train_frequency: int = 50,
    num_players: int = 2,
):
    """Pool-based training loop for Mazerush with multiple agents."""
    n = len(agents)
    ep_counts = [0] * n
    win_counts = [0] * n
    draw_counts = [0] * n
    lose_counts = [0] * n
    recent_win: list[collections.deque] = [collections.deque(maxlen=1000) for _ in range(n)]
    recent_draw: list[collections.deque] = [collections.deque(maxlen=1000) for _ in range(n)]
    recent_lost: list[collections.deque] = [collections.deque(maxlen=1000) for _ in range(n)]
    recent_losses: list[collections.deque] = [collections.deque(maxlen=1000) for _ in range(n)]
    recent_rewards: list[collections.deque] = [collections.deque(maxlen=1000) for _ in range(n)]

    for ep in range(num_episodes):
        # Sample a pair of pool indices
        if self_play:
            idxs = random.choices(range(n), k=num_players)
        else:
            idxs = random.sample(range(n), num_players)

        pair = [agents[idx] for idx in idxs]
        per_player_steps, cumulative_rewards, results = run_episode(pair, env, train=True)

        # Register steps & update per-agent stats
        for i, pool_idx in enumerate(idxs):
            agent = agents[pool_idx]
            if not isinstance(agent, (DeepQAgent, PPOAgent)):
                continue
            agent.register_action_steps(per_player_steps[i])
            ep_counts[pool_idx] += 1
            recent_win[pool_idx].append(1 if results[i] == "win" else 0)
            recent_draw[pool_idx].append(1 if results[i] == "draw" else 0)
            recent_lost[pool_idx].append(1 if results[i] == "lose" else 0)
            recent_rewards[pool_idx].append(cumulative_rewards[i])
            win_counts[pool_idx] += 1 if results[i] == "win" else 0
            draw_counts[pool_idx] += 1 if results[i] == "draw" else 0
            lose_counts[pool_idx] += 1 if results[i] == "lose" else 0

            # Train when this agent's own ep count hits a multiple of train_frequency
            if ep_counts[pool_idx] % train_frequency == 0:
                loss_hist = agent.train()
                if loss_hist:
                    recent_losses[pool_idx].extend(loss_hist)

                wr = sum(recent_win[pool_idx]) / max(len(recent_win[pool_idx]), 1) * 100
                dr = sum(recent_draw[pool_idx]) / max(len(recent_draw[pool_idx]), 1) * 100
                lr = sum(recent_lost[pool_idx]) / max(len(recent_lost[pool_idx]), 1) * 100
                avg_reward = sum(recent_rewards[pool_idx]) / max(len(recent_rewards[pool_idx]), 1)
                avg_loss = statistics.mean(recent_losses[pool_idx]) if recent_losses[pool_idx] else 0.0

                print(
                    f"[Player {pool_idx}] Ep {ep+1} (own: {ep_counts[pool_idx]}) "
                    f"- Loss: {avg_loss:.6f} "
                    f"- Win Rate: {wr:.1f}% - Draw Rate: {dr:.1f}% - Loss Rate: {lr:.1f}% "
                    f"- Avg Reward: {avg_reward:.2f} "
                    f"- Total Wins: {win_counts[pool_idx]} "
                    f"- Total Draws: {draw_counts[pool_idx]} "
                    f"- Total Losses: {lose_counts[pool_idx]} "
                    + (f"- Eps: {agent._epsilon:.4f}" if isinstance(agent, DeepQAgent) else "")
                )

        # Checkpoint all pool agents at global save_interval
        if checkpoint_dir and (ep + 1) % save_interval == 0:
            for i, agent in enumerate(agents):
                if not isinstance(agent, (DeepQAgent, PPOAgent)):
                    continue
                path = os.path.join(checkpoint_dir, f'agent{i}_ckpt_{ep + 1}.pt')
                agent.save(path)
                print(f'Saved agent {i} checkpoint to {path}')

    print(f"\nPool training complete.")
    for i in range(n):
        if not isinstance(agents[i], (DeepQAgent, PPOAgent)):
            continue
        print(f"  Agent {i}: {ep_counts[i]} episodes, {win_counts[i]}W / {draw_counts[i]}D / {lose_counts[i]}L")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main function to run the Mazerush game."""

    parser = argparse.ArgumentParser(description="Mazerush")
    parser.add_argument("--config", type=str, default="config/mazerush.yaml", help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Mode: train or test")
    parser.add_argument("--render", type=str, default=None, help="Render mode (human, human_full, None)")
    parser.add_argument("--episodes-override", type=int, default=None, help="Override num_episodes for quick testing")
    parser.add_argument(
        "--self-play",
        action="store_true",
        default=False,
        help="Allow self-play in pool training (same agent can fill both slots).",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    env_config = config["env"].get("config") or {}
    player_configs = config.get("players", [{"type": "RandomAgent"}, {"type": "RandomAgent"}])

    render_mode = args.render
    if render_mode is None and args.mode == "test":
        render_mode = "human"
    env = MazerushEnv(render_mode=render_mode, **env_config)

    num_states = env.observation_space.shape[0]
    agent_config = config.get("agent", {})
    ppo_agent_config = config.get("ppo_agent", {})
    agents = _build_agents(player_configs, env.action_space, num_states, agent_config, ppo_agent_config)

    # Resume DeepQAgents
    resume_paths = args.resume or config.get("training", {}).get("resume_paths")
    if resume_paths:
        resume_paths = resume_paths.split(",")
        resume_idx = 0
        for i, agent in enumerate(agents):
            if isinstance(agent, (DeepQAgent, PPOAgent)):
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

        train(
            agents, env,
            num_episodes=num_episodes,
            self_play=args.self_play,
            checkpoint_dir=checkpoint_dir,
            save_interval=training_cfg.get("save_interval", 1000),
            train_frequency=training_cfg.get("train_frequency", 50),
        )
    else:
        # Test / play mode
        print("Starting Mazerush in play mode. Close the window to exit.")
        while True:
            per_player_steps, rewards, results = run_episode(agents, env, train=False)
            print(f"Episode finished. Rewards: {rewards}, Results: {results}")
            if not pygame.display.get_init():
                break

    env.close()

if __name__ == "__main__":
    main()
