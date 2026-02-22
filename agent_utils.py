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
        learning_rate: float = 1e-4,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.99,
        replay_buffer_episodes: int = 1000,
        train_batch_size: int = 128,
        train_epochs: int = 1000,
        hidden_size: int = 64,
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
        self._replay_buffer = collections.deque(maxlen=replay_buffer_episodes)
        self._action_value_fn = _DeepQNetwork(self._num_states, hidden_size, action_space.n).to(self._device)
        self._optimizer = torch.optim.Adam(self._action_value_fn.parameters(), lr=self._learning_rate)

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
        tensor_steps = []
        for step in action_steps:
            tensor_step = dataclasses.replace(
                step,
                state=torch.tensor(step.state, dtype=torch.float32, device=self._device),
                state_next=torch.tensor(step.state_next, dtype=torch.float32, device=self._device)
            )
            tensor_steps.append(tensor_step)
        self._replay_buffer.append(tensor_steps)
    
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
            actions = torch.tensor([step.action for step in batch], device=self._device)
            rewards = torch.tensor([step.reward for step in batch], dtype=torch.float32, device=self._device)
            dones = torch.tensor([step.done for step in batch], dtype=torch.float32, device=self._device)

            self._optimizer.zero_grad()
            
            # Current Q values for the selected actions
            # We need to gather the Q-values corresponding to the taken actions
            # Output of _predict_rewards(states) is [batch_size, num_actions]
            # We want to select the Q-value for the action taken at each step
            state_tensors = torch.stack([step.state for step in batch])
            next_state_tensors = torch.stack([step.state_next for step in batch])

            q_values = self._action_value_fn(state_tensors)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Target Q values
            with torch.no_grad():
                next_q_values = self._action_value_fn(next_state_tensors)
                max_next_q_values = next_q_values.max(1)[0]
                # If done, target is just reward. Else reward + discount * max_next_q
                targets = rewards + self._discount_factor * max_next_q_values * (1 - dones)
            
            loss = torch.nn.functional.mse_loss(q_values, targets)
            loss.backward()
            self._optimizer.step()
            loss_history.append(loss.item())
        
        # Decay epsilon after each training episode
        self._epsilon = max(self._epsilon_end, self._epsilon * self._epsilon_decay)
        return loss_history

    def save(self, path: str) -> None:
        torch.save({
            'model_state_dict': self._action_value_fn.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'epsilon': self._epsilon
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path)
        self._action_value_fn.load_state_dict(checkpoint['model_state_dict'])
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
        self._action_queue: collections.deque[int] = collections.deque(maxlen=8)

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
        if self._action_queue:
            return self._action_queue.popleft()
        return 5  # ACTION_NOTHING


class _DeepQNetwork(torch.nn.Module):
    def __init__(self, num_states: int, hidden_size: int, num_actions: int):
        super().__init__()
        self.num_states = num_states
        
        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(num_states, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # state is [batch_size, num_states]
        # output shape is [batch_size, num_actions]
        return self.output_net(state)
