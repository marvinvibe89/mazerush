import abc
import collections
import dataclasses
import random
import statistics
from typing import Sequence

import numpy as np
import pygame
import torch
from typing_extensions import override

from mazerush_utils import Player


@dataclasses.dataclass(frozen=True)
class ActionStep:
    action: int
    state: int
    state_next: int
    reward: float
    done: bool


class Agent(abc.ABC):

    def __init__(self, action_space):
        self._action_space = action_space

    @abc.abstractmethod
    def select_action(self, state, train: bool = False) -> int:
        """Select an action given a state (int or np.ndarray)."""
        pass

    def set_player(self, player: Player):
        """Link this agent to a player object for internal status checks."""
        pass


class RandomAgent(Agent):

    @override
    def select_action(self, state, train: bool = False) -> int:
        return self._action_space.sample()


class NothingAgent(Agent):

    @override
    def select_action(self, state, train: bool = False) -> int:
        return 5  # ACTION_NOTHING


class DeepQAgent(Agent):

    def __init__(
        self,
        action_space,
        num_states: int,
        *,
        learning_rate: float,
        discount_factor: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        replay_buffer_episodes: int,
        train_batch_size: int,
        train_epochs: int,
        hidden_size: int,
        tau: float,
    ):
        super().__init__(action_space)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._num_states = num_states
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._epsilon = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay = epsilon_decay
        self._train_batch_size = train_batch_size
        self._train_epochs = train_epochs
        self._tau = tau
        self._replay_buffer = collections.deque(maxlen=replay_buffer_episodes)
        self._action_value_fn = _DeepQNetwork(
            self._num_states, hidden_size, int(action_space.n)
        ).to(self._device)
        self._target_net = _DeepQNetwork(
            self._num_states, hidden_size, int(action_space.n)
        ).to(self._device)
        self._target_net.load_state_dict(self._action_value_fn.state_dict())
        for param in self._target_net.parameters():
            param.requires_grad = False
        self._optimizer = torch.optim.Adam(
            self._action_value_fn.parameters(), lr=self._learning_rate
        )

    def _state_to_tensor(self, state) -> torch.Tensor:
        """Convert a state float array to a tensor."""
        return torch.tensor(state, dtype=torch.float32, device=self._device)

    def _predict_rewards(self, state) -> torch.Tensor:
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=self._device)
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Add batch dimension
            return self._action_value_fn(x)

    def select_action(self, state, train: bool = False) -> int:
        if train and random.random() < self._epsilon:
            return self._action_space.sample()
        return torch.argmax(self._predict_rewards(state)[0]).item()

    def register_action_steps(self, action_steps: list[ActionStep]):
        np_steps = []
        for i, step in enumerate(action_steps):
            np_step = dataclasses.replace(
                step,
                reward=step.reward,
                state=np.asarray(step.state, dtype=np.float32),
                state_next=np.asarray(step.state_next, dtype=np.float32),
            )
            np_steps.append(np_step)
        self._replay_buffer.append(np_steps)
        # Decay epsilon after each episode registered
        self._epsilon = max(self._epsilon_end, self._epsilon * self._epsilon_decay)

    def train(self) -> Sequence[float]:
        # Flatten the replay buffer to get a list of all steps
        all_steps = [step for episode in self._replay_buffer for step in episode]
        if len(all_steps) < self._train_batch_size:
            return []

        loss_history = []
        for epoch in range(self._train_epochs):
            # Sample a batch of transitions
            batch = random.sample(all_steps, self._train_batch_size)
            # Prepare batch data
            actions = torch.tensor(
                [step.action for step in batch], device=self._device
            )
            rewards = torch.tensor(
                [step.reward for step in batch],
                dtype=torch.float32,
                device=self._device,
            )
            dones = torch.tensor(
                [step.done for step in batch],
                dtype=torch.float32,
                device=self._device,
            )

            self._optimizer.zero_grad()
            # Current Q values for the selected actions
            # We need to gather the Q-values corresponding to the taken actions
            # Output of _predict_rewards(states) is [batch_size, num_actions]
            # We want to select the Q-value for the action taken at each step
            state_tensors = torch.tensor(
                np.stack([step.state for step in batch]),
                dtype=torch.float32,
                device=self._device,
            )
            next_state_tensors = torch.tensor(
                np.stack([step.state_next for step in batch]),
                dtype=torch.float32,
                device=self._device,
            )

            q_values = self._action_value_fn(state_tensors)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Target Q values (Double DQN logic)
            with torch.no_grad():
                next_q_online = self._action_value_fn(next_state_tensors)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q_target = self._target_net(next_state_tensors)
                target_q_values = next_q_target.gather(1, next_actions).squeeze(1)
                targets = rewards + self._discount_factor * target_q_values * (
                    1 - dones
                )

            loss = torch.nn.functional.mse_loss(q_values, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._action_value_fn.parameters(), max_norm=1.0
            )
            self._optimizer.step()
            loss_history.append(loss.item())

            # Soft update of target network inside the training loop
            with torch.no_grad():
                for target_param, param in zip(
                    self._target_net.parameters(), self._action_value_fn.parameters()
                ):
                    target_param.data.copy_(
                        self._tau * param.data + (1.0 - self._tau) * target_param.data
                    )

        return loss_history

    def save(self, path: str) -> None:
        torch.save({
            'model_state_dict': self._action_value_fn.state_dict(),
            'target_state_dict': self._target_net.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'epsilon': self._epsilon
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self._device)
        self._action_value_fn.load_state_dict(checkpoint['model_state_dict'])
        self._target_net.load_state_dict(checkpoint['target_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._epsilon = checkpoint['epsilon']


class PPOAgent(Agent):

    def __init__(
        self,
        action_space,
        num_states: int,
        *,
        learning_rate: float,
        discount_factor: float,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        train_epochs: int = 4,
        train_batch_size: int = 64,
        hidden_size: int = 256,
        max_grad_norm: float = 0.5,
    ):
        super().__init__(action_space)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._num_states = num_states
        self._discount_factor = discount_factor
        self._gae_lambda = gae_lambda
        self._clip_epsilon = clip_epsilon
        self._entropy_coef = entropy_coef
        self._value_loss_coef = value_loss_coef
        self._train_epochs = train_epochs
        self._train_batch_size = train_batch_size
        self._max_grad_norm = max_grad_norm

        self._actor_critic = _PPOActorCritic(
            num_states, hidden_size, int(action_space.n)
        ).to(self._device)
        self._optimizer = torch.optim.Adam(
            self._actor_critic.parameters(), lr=learning_rate
        )

        # Rollout buffers — filled during select_action, consumed by train.
        self._rollout_log_probs: list[torch.Tensor] = []
        self._rollout_values: list[torch.Tensor] = []
        # Episode buffer — list of episodes, each a list of ActionStep.
        self._episode_buffer: list[list[ActionStep]] = []
        self._episode_log_probs: list[list[torch.Tensor]] = []
        self._episode_values: list[list[torch.Tensor]] = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state, train: bool = False) -> int:
        state_t = torch.tensor(state, dtype=torch.float32, device=self._device)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)

        if train:
            logits, value = self._actor_critic(state_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            self._rollout_log_probs.append(dist.log_prob(action))
            self._rollout_values.append(value.squeeze(-1))
            return action.item()
        else:
            with torch.no_grad():
                logits, _ = self._actor_critic(state_t)
            return logits.argmax(dim=-1).item()

    # ------------------------------------------------------------------
    # Rollout storage
    # ------------------------------------------------------------------

    def register_action_steps(self, action_steps: list[ActionStep]):
        """Store one episode of transitions along with collected log_probs/values."""
        self._episode_buffer.append(action_steps)
        self._episode_log_probs.append(list(self._rollout_log_probs))
        self._episode_values.append(list(self._rollout_values))
        # Clear per-step buffers for the next episode.
        self._rollout_log_probs = []
        self._rollout_values = []

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def _compute_gae(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute GAE advantages across all stored episodes.

        Returns flat tensors: (states, actions, old_log_probs, advantages, returns).
        """
        all_states, all_actions, all_log_probs = [], [], []
        all_advantages, all_returns = [], []

        for ep_idx, episode in enumerate(self._episode_buffer):
            log_probs = self._episode_log_probs[ep_idx]
            values = self._episode_values[ep_idx]

            T = len(episode)
            if T == 0:
                continue

            rewards = torch.tensor(
                [step.reward for step in episode],
                dtype=torch.float32, device=self._device,
            )
            dones = torch.tensor(
                [step.done for step in episode],
                dtype=torch.float32, device=self._device,
            )
            vals = torch.cat(values)  # (T,)

            # Bootstrap value for the last step.
            if episode[-1].done:
                next_value = torch.tensor(0.0, device=self._device)
            else:
                with torch.no_grad():
                    s = torch.tensor(
                        episode[-1].state_next,
                        dtype=torch.float32, device=self._device,
                    ).unsqueeze(0)
                    _, nv = self._actor_critic(s)
                    next_value = nv.squeeze()

            advantages = torch.zeros(T, device=self._device)
            gae = torch.tensor(0.0, device=self._device)
            for t in reversed(range(T)):
                nv = next_value if t == T - 1 else vals[t + 1]
                delta = rewards[t] + self._discount_factor * nv * (1 - dones[t]) - vals[t]
                gae = delta + self._discount_factor * self._gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae

            returns = advantages + vals.detach()

            all_states.extend(
                np.asarray(step.state, dtype=np.float32) for step in episode
            )
            all_actions.extend(step.action for step in episode)
            all_log_probs.append(torch.cat(log_probs))  # already tensors
            all_advantages.append(advantages)
            all_returns.append(returns)

        states_t = torch.tensor(
            np.stack(all_states), dtype=torch.float32, device=self._device
        )
        actions_t = torch.tensor(all_actions, dtype=torch.long, device=self._device)
        old_log_probs_t = torch.cat(all_log_probs).detach()
        advantages_t = torch.cat(all_advantages).detach()
        returns_t = torch.cat(all_returns).detach()

        # Normalise advantages.
        if advantages_t.numel() > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        return states_t, actions_t, old_log_probs_t, advantages_t, returns_t

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> list[float]:
        if not self._episode_buffer:
            return []

        states, actions, old_log_probs, advantages, returns = self._compute_gae()
        N = states.size(0)
        if N == 0:
            self._episode_buffer.clear()
            self._episode_log_probs.clear()
            self._episode_values.clear()
            return []

        loss_history: list[float] = []
        for _ in range(self._train_epochs):
            indices = torch.randperm(N, device=self._device)
            for start in range(0, N, self._train_batch_size):
                idx = indices[start : start + self._train_batch_size]

                logits, values = self._actor_critic(states[idx])
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()

                # Clipped surrogate objective.
                ratio = (new_log_probs - old_log_probs[idx]).exp()
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1.0 - self._clip_epsilon, 1.0 + self._clip_epsilon) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = torch.nn.functional.mse_loss(values.squeeze(-1), returns[idx])

                loss = policy_loss + self._value_loss_coef * value_loss - self._entropy_coef * entropy

                self._optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._actor_critic.parameters(), max_norm=self._max_grad_norm
                )
                self._optimizer.step()
                loss_history.append(loss.item())

        # Clear on-policy data after training.
        self._episode_buffer.clear()
        self._episode_log_probs.clear()
        self._episode_values.clear()
        return loss_history

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            'model_state_dict': self._actor_critic.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self._device)
        self._actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class HumanAgent(Agent):
    """Agent controlled by keyboard input with an action queue.

    Key presses are buffered so that inputs arriving before a cooldown
    lifts are executed on the exact tick the cooldown expires rather than
    being dropped.
    """

    # Action mapping (pygame key constants)
    _KEY_MAP: dict[int, int] | None = None

    def __init__(self, action_space):
        super().__init__(action_space)
        self._player: Player | None = None
        self._action_queue: collections.deque[int] = collections.deque(maxlen=8)

    @override
    def set_player(self, player: Player):
        self._player = player

    @staticmethod
    def _get_key_map() -> dict[int, int]:
        """Get the key map for the human agent."""
        return {
            pygame.K_UP: 0,     # ACTION_UP
            pygame.K_DOWN: 1,   # ACTION_DOWN
            pygame.K_LEFT: 2,   # ACTION_LEFT
            pygame.K_RIGHT: 3,  # ACTION_RIGHT
            pygame.K_SPACE: 4,  # ACTION_SHOOT
        }

    def key_listener(self, event) -> None:
        """Called by the renderer / run-loop with each pygame event."""
        if event.type == pygame.KEYDOWN:
            key_map = self._get_key_map()
            if event.key in key_map:
                self._action_queue.append(key_map[event.key])

    @override
    def select_action(self, state, train: bool = False) -> int:
        # Respect cooldown by checking the player object directly if available.
        can_act = True
        if self._player is not None:
            can_act = (self._player.move_cooldown_remaining == 0)

        if self._action_queue:
            # Do not wait for cooldown to shoot.
            if can_act or self._action_queue[0] == 4:
                return self._action_queue.popleft()
        return 5  # ACTION_NOTHING


class _DeepQNetwork(torch.nn.Module):
    def __init__(self, num_states: int, hidden_size: int, num_actions: int):
        super().__init__()
        self.feature_layer = torch.nn.Sequential(
            torch.nn.Linear(num_states, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
        )
        self.value_stream = torch.nn.Linear(hidden_size, 1)
        self.advantage_stream = torch.nn.Linear(hidden_size, num_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class _PPOActorCritic(torch.nn.Module):
    def __init__(self, num_states: int, hidden_size: int, num_actions: int):
        super().__init__()
        self.feature_layer = torch.nn.Sequential(
            torch.nn.Linear(num_states, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
        )
        self.actor_head = torch.nn.Linear(hidden_size, num_actions)  # logits
        self.critic_head = torch.nn.Linear(hidden_size, 1)           # value

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_layer(state)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value


def build_agents(
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
