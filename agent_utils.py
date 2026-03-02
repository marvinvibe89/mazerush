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

from mazerush_utils import Player, NUM_OBS_CELL_TYPES


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
        obs_shape: tuple[int, ...],
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
        self._obs_shape = obs_shape
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
            obs_shape, hidden_size, int(action_space.n)
        ).to(self._device)
        self._target_net = _DeepQNetwork(
            obs_shape, hidden_size, int(action_space.n)
        ).to(self._device)
        self._target_net.load_state_dict(self._action_value_fn.state_dict())
        for param in self._target_net.parameters():
            param.requires_grad = False
        self._optimizer = torch.optim.Adam(
            self._action_value_fn.parameters(), lr=self._learning_rate
        )

    def _state_to_tensor(self, state) -> torch.Tensor:
        arr = np.asarray(state, dtype=np.float32) / NUM_OBS_CELL_TYPES
        return torch.tensor(arr, dtype=torch.float32, device=self._device)

    def _predict_rewards(self, state) -> torch.Tensor:
        with torch.no_grad():
            x = self._state_to_tensor(state)
            if x.dim() == len(self._obs_shape):
                x = x.unsqueeze(0)
            return self._action_value_fn(x)

    def select_action(self, state, train: bool = False) -> int:
        if train and random.random() < self._epsilon:
            return self._action_space.sample()
        return torch.argmax(self._predict_rewards(state)[0]).item()

    def register_action_steps(self, action_steps: list[ActionStep]):
        np_steps = []
        for step in action_steps:
            np_step = dataclasses.replace(
                step,
                state=np.asarray(step.state, dtype=np.int8),
                state_next=np.asarray(step.state_next, dtype=np.int8),
            )
            np_steps.append(np_step)
        self._replay_buffer.append(np_steps)

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
            states_np = (
                    np.stack([step.state for step in batch]).astype(np.float32)
                    / NUM_OBS_CELL_TYPES
            )
            next_states_np = (
                np.stack([step.state_next for step in batch]).astype(np.float32)
                / NUM_OBS_CELL_TYPES
            )
            state_tensors = torch.tensor(
                states_np, dtype=torch.float32, device=self._device
            )
            next_state_tensors = torch.tensor(
                next_states_np, dtype=torch.float32, device=self._device
            )

            q_values = self._action_value_fn(state_tensors)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Target Q values (Double DQN logic)
            with torch.no_grad():
                next_q_online = self._action_value_fn(next_state_tensors)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q_target = self._target_net(next_state_tensors)
                max_next_q_values = next_q_target.gather(1, next_actions).squeeze(1)
                targets = rewards + self._discount_factor * max_next_q_values * (
                    1 - dones
                )

            loss = torch.nn.functional.smooth_l1_loss(q_values, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._action_value_fn.parameters(), max_norm=1.0
            )
            self._optimizer.step()
            loss_history.append(loss.item())

            # Soft update of target network inside epoch loop
            for target_param, param in zip(
                self._target_net.parameters(), self._action_value_fn.parameters()
            ):
                target_param.data.copy_(
                    self._tau * param.data + (1.0 - self._tau) * target_param.data
                )

        # Decay epsilon after each training episode
        self._epsilon = max(self._epsilon_end, self._epsilon * self._epsilon_decay)
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
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        hidden_size: int,
        num_actions: int,
    ):
        super().__init__()
        in_channels = obs_shape[0]
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool2d((4, 4)),
            torch.nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            conv_out_size = self.conv(dummy).shape[1]

        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(conv_out_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
        )
        self.advantage_head = torch.nn.Sequential(
            torch.nn.Linear(conv_out_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.conv(state)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


def build_agents(
    player_configs: list[dict],
    action_space,
    num_states: int,
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
