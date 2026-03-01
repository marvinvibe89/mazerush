import pygame
from agent_utils import HumanAgent
from mazerush_utils import Player

class MockActionSpace:
    def __init__(self, n):
        self.n = n

def test_human_agent_buffering_with_player():
    action_space = MockActionSpace(6)
    agent = HumanAgent(action_space)
    player = Player(10, 10)
    agent.set_player(player)
    
    # Add some actions to the queue
    # Simulate UP (0) then RIGHT (3)
    event_up = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP)
    event_right = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT)
    agent.key_listener(event_up)
    agent.key_listener(event_right)
    
    assert len(agent._action_queue) == 2
    
    # Simulate player in COOLDOWN
    player.move_cooldown_remaining = 5
    
    # Should return NOTHING (5) and NOT pop
    action = agent.select_action(None) # state is ignored now
    assert action == 5
    assert len(agent._action_queue) == 2
    
    # Simulate player READY (cooldown 0)
    player.move_cooldown_remaining = 0
    
    # Should return UP (0) and pop
    action = agent.select_action(None)
    assert action == 0
    assert len(agent._action_queue) == 1
    
    # Next call should return RIGHT (3)
    action = agent.select_action(None)
    assert action == 3
    assert len(agent._action_queue) == 0
    
    # Then NOTHING (5)
    action = agent.select_action(None)
    assert action == 5

    print("Refactored Test passed!")

if __name__ == "__main__":
    test_human_agent_buffering_with_player()
