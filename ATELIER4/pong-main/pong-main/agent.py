"""
Implementation of Q-Learning algorithm for controlling the player paddle in Pong.
"""

import numpy as np
import os
import pickle
import collections
import random

class Qlearning:
    """
    Q-learning Agent for Pong game.
    
    The agent learns to control the player paddle using the Q-learning algorithm
    with epsilon-greedy action selection.
    
    Parameters
    ----------
    alpha : float
        Learning rate (0 < alpha <= 1). Controls how much new information 
        overrides old information.
    gamma : float
        Discount factor (0 <= gamma <= 1). Determines the importance of 
        future rewards.
    eps : float
        Epsilon for epsilon-greedy action selection. Probability of choosing 
        a random action instead of the greedy action.
    eps_decay : float
        Epsilon decay rate. After each episode, epsilon is multiplied by 
        (1 - eps_decay). Larger value = faster decay.
    
    Attributes
    ----------
    Q : collections.defaultdict
        Q-table storing state-action values. Keys are (state, action) tuples.
    actions : list
        List of possible actions: [-1 (up), 0 (stay), 1 (down)]
    rewards_history : list
        History of rewards received during training.
    """

    def __init__(self, alpha=0.5, gamma=0.9, eps=0.1, eps_decay=0.):
        # Agent parameters
        self.alpha = alpha          # Learning rate
        self.gamma = gamma          # Discount factor
        self.eps = eps              # Epsilon for exploration
        self.eps_decay = eps_decay  # Epsilon decay rate
        self.initial_eps = eps      # Store initial epsilon for reset
        
        # Q-table: defaultdict returns 0 for unseen state-action pairs
        self.Q = collections.defaultdict(float)
        
        # Possible actions: -1 = move up, 0 = stay, 1 = move down
        self.actions = [-1, 0, 1]
        
        # Training history
        self.rewards_history = []
        self.episode_rewards = 0
        
    def get_Q_value(self, state, action):
        """
        Get Q-value for a state-action pair.
        
        Parameters
        ----------
        state : str
            State representation
        action : int
            Action (-1, 0, or 1)
            
        Returns
        -------
        float
            Q-value for the state-action pair
        """
        return self.Q[(state, action)]
    
    def get_max_Q(self, state):
        """
        Get the maximum Q-value for a given state across all actions.
        
        Parameters
        ----------
        state : str
            State representation
            
        Returns
        -------
        float
            Maximum Q-value for the state
        """
        q_values = [self.get_Q_value(state, a) for a in self.actions]
        return max(q_values)
    
    def get_action(self, state):
        """
        Select action using epsilon-greedy policy.
        
        With probability epsilon, choose a random action (exploration).
        Otherwise, choose the action with highest Q-value (exploitation).
        This is the Greedy Choose algorithm as specified in the assignment.
        
        Parameters
        ----------
        state : str
            Current state representation
            
        Returns
        -------
        int
            Selected action (-1, 0, or 1)
        """
        # Epsilon-greedy action selection
        if random.random() < self.eps:
            # Exploration: random action
            return random.choice(self.actions)
        else:
            # Exploitation: greedy action (best Q-value)
            q_values = [self.get_Q_value(state, a) for a in self.actions]
            max_q = max(q_values)
            
            # If multiple actions have the same max Q-value, choose randomly among them
            best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
            return random.choice(best_actions)
    
    def update(self, s, s_, a, a_, r):
        """
        Perform the Q-Learning update of Q values.
        
        Q-Learning update rule:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a'(Q(s', a')) - Q(s, a)]
        
        Note: Q-learning is off-policy, so a_ (next action) is NOT used.
        We use the max Q-value of the next state instead.
        
        Parameters
        ----------
        s : str
            Previous state
        s_ : str
            New state
        a : int
            Previous action taken
        a_ : int
            New action (NOT used in Q-learning, included for API compatibility)
        r : float
            Reward received after executing action "a" in state "s"
        """
        # Current Q-value
        current_q = self.get_Q_value(s, a)
        
        # Maximum Q-value for next state (Q-learning uses max, not actual next action)
        max_next_q = self.get_max_Q(s_)
        
        # Q-learning update rule
        # Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        new_q = current_q + self.alpha * (r + self.gamma * max_next_q - current_q)
        
        # Update Q-table
        self.Q[(s, a)] = new_q
        
        # Track rewards
        self.episode_rewards += r
    
    def end_episode(self):
        """
        Called at the end of each episode to update statistics and decay epsilon.
        """
        # Store episode reward
        self.rewards_history.append(self.episode_rewards)
        self.episode_rewards = 0
        
        # Decay epsilon
        self.eps = self.eps * (1 - self.eps_decay)
        # Ensure epsilon doesn't go below a minimum threshold
        self.eps = max(self.eps, 0.01)
    
    def reset_episode_reward(self):
        """Reset the episode reward counter."""
        self.episode_rewards = 0
    
    def add_reward(self, reward):
        """Add reward to current episode total."""
        self.episode_rewards += reward
    
    def save(self, filepath):
        """
        Save the Q-table to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the Q-table
        """
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.Q), f)
        print(f"Q-table saved to {filepath}")
    
    def load(self, filepath):
        """
        Load a Q-table from a file.
        
        Parameters
        ----------
        filepath : str
            Path to the saved Q-table
        """
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.Q = collections.defaultdict(float, pickle.load(f))
            print(f"Q-table loaded from {filepath}")
        else:
            print(f"No saved Q-table found at {filepath}")
    
    def get_stats(self):
        """
        Get training statistics.
        
        Returns
        -------
        dict
            Dictionary containing training statistics
        """
        return {
            'total_episodes': len(self.rewards_history),
            'total_states': len(set(s for s, a in self.Q.keys())),
            'average_reward': np.mean(self.rewards_history) if self.rewards_history else 0,
            'current_epsilon': self.eps,
            'q_table_size': len(self.Q)
        }
    
    def __str__(self):
        stats = self.get_stats()
        return (f"Q-Learning Agent:\n"
                f"  Alpha (learning rate): {self.alpha}\n"
                f"  Gamma (discount): {self.gamma}\n"
                f"  Epsilon: {self.eps:.4f}\n"
                f"  Episodes trained: {stats['total_episodes']}\n"
                f"  Q-table size: {stats['q_table_size']}\n"
                f"  Unique states: {stats['total_states']}")
