import abc
import dataclasses
import random
import statistics
import collections
import torch
from typing import Sequence
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
    def select_action(self, state: int, train: bool = False) -> int:
        pass


class RandomAgent(Agent):

    @override
    def select_action(self, state: int, train: bool = False) -> int:
        return self._action_space.sample()


class DeepQAgent(Agent):

    def __init__(
        self,
        action_space,
        num_states: int,
        learning_rate: float,
        discount_factor:float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        replay_buffer_episodes: int,
        train_batch_size: int,
        train_epochs: int,
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
        self._action_value_fn = torch.nn.Sequential(
            torch.nn.Linear(num_states, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, action_space.n),
        ).to(self._device)
        self._optimizer = torch.optim.Adam(self._action_value_fn.parameters(), lr=self._learning_rate)
    
    def _predict_rewards(self, state: int) -> torch.Tensor:
        return self._action_value_fn(torch.nn.functional.one_hot(torch.tensor(state, device=self._device), num_classes=self._num_states).float())

    def select_action(self, state: int, train: bool = False) -> int:
        if train and random.random() < self._epsilon:
            return self._action_space.sample()
        return torch.argmax(self._predict_rewards(state)).item()
    
    def register_action_steps(self, action_steps: list[ActionStep]):
        self._replay_buffer.append(action_steps)
    
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
            states = torch.tensor([step.state for step in batch], device=self._device)
            actions = torch.tensor([step.action for step in batch], device=self._device)
            rewards = torch.tensor([step.reward for step in batch], dtype=torch.float32, device=self._device)
            next_states = torch.tensor([step.state_next for step in batch], device=self._device)
            dones = torch.tensor([step.done for step in batch], dtype=torch.float32, device=self._device)

            self._optimizer.zero_grad()
            
            # Current Q values for the selected actions
            # We need to gather the Q-values corresponding to the taken actions
            # Output of _predict_rewards(states) is [batch_size, num_actions]
            # We want to select the Q-value for the action taken at each step
            q_values = self._action_value_fn(torch.nn.functional.one_hot(states, num_classes=self._num_states).float())
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Target Q values
            with torch.no_grad():
                next_q_values = self._action_value_fn(torch.nn.functional.one_hot(next_states, num_classes=self._num_states).float())
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
