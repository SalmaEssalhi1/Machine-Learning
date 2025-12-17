"""
Main training script for Q-Learning Pong agents.

Supports three training modes:
1. rl_vs_ai: Q-Learning agent vs Simple AI
2. rl_vs_human: Q-Learning agent vs Human player
3. rl_vs_rl: Q-Learning agent vs Q-Learning agent
"""

import game as g
import agent as ag
import matplotlib.pyplot as plt
import numpy as np
import os
import time


def plot_agent_reward(rewards, title="Agent Cumulative Reward vs. Episode", save_path=None):
    """
    Function to plot agent's accumulated reward vs. iteration.
    
    Parameters
    ----------
    rewards : list
        List of rewards per episode
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Cumulative reward
    axes[0].plot(np.cumsum(rewards), 'b-', linewidth=1)
    axes[0].set_title('Cumulative Reward vs. Episode')
    axes[0].set_ylabel('Cumulative Reward')
    axes[0].set_xlabel('Episode')
    axes[0].grid(True, alpha=0.3)
    
    # Moving average reward
    window = min(100, len(rewards) // 10 + 1)
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[1].plot(moving_avg, 'g-', linewidth=1)
        axes[1].set_title(f'Moving Average Reward (window={window})')
    else:
        axes[1].plot(rewards, 'g-', linewidth=1)
        axes[1].set_title('Reward per Episode')
    axes[1].set_ylabel('Reward')
    axes[1].set_xlabel('Episode')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_training_comparison(results_dict, save_path=None):
    """
    Compare training results from different modes.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with mode names as keys and reward lists as values
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['blue', 'green', 'red']
    
    # Cumulative rewards comparison
    for i, (mode, rewards) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        axes[0].plot(np.cumsum(rewards), color=color, label=mode, linewidth=1)
    
    axes[0].set_title('Cumulative Reward Comparison')
    axes[0].set_ylabel('Cumulative Reward')
    axes[0].set_xlabel('Episode')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Final performance bar chart
    final_rewards = [np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards) 
                     for rewards in results_dict.values()]
    modes = list(results_dict.keys())
    
    bars = axes[1].bar(modes, final_rewards, color=colors[:len(modes)])
    axes[1].set_title('Average Reward (Last 100 Episodes)')
    axes[1].set_ylabel('Average Reward')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, final_rewards):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()


class GameLearning:
    """
    Main class for training Q-Learning agents in Pong.
    
    Parameters
    ----------
    alpha : float
        Learning rate
    gamma : float
        Discount factor
    epsilon : float
        Initial exploration rate
    eps_decay : float
        Epsilon decay rate
    """
    
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1, eps_decay=0.001):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        
        # Create directories for saving results
        self.models_dir = "models"
        self.graphe_dir = "graphe"
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.graphe_dir, exist_ok=True)
        
        self.games_played = 0
        self.all_rewards = {}


    def train_rl_vs_ai(self, episodes=1000, render=False, save_interval=100):
        """
        Train Q-Learning agent against simple AI opponent.
        
        Mode: Agent RL vs Agent AI
        
        Parameters
        ----------
        episodes : int
            Number of training episodes
        render : bool
            Whether to render the game
        save_interval : int
            Save Q-table every N episodes
            
        Returns
        -------
        tuple
            (agent, rewards_history)
        """
        print("=" * 60)
        print("MODE: Agent RL vs Agent AI")
        print("=" * 60)
        
        agent = ag.Qlearning(self.alpha, self.gamma, self.epsilon, self.eps_decay)
        game = g.Game(agent, mode='rl_vs_ai', render=render)
        
        rewards = []
        wins = 0
        losses = 0
        
        for episode in range(episodes):
            state = game.reset()
            episode_reward = 0
            done = False
            steps = 0
            max_steps = 5000  # Prevent infinite episodes
            
            while not done and steps < max_steps:
                # Get action from agent
                action = agent.get_action(state)
                
                # Execute action
                new_state, reward, done = game.step(action)
                
                # Update Q-table
                new_action = agent.get_action(new_state)
                agent.update(state, new_state, action, new_action, reward)
                
                episode_reward += reward
                state = new_state
                steps += 1
            
            rewards.append(episode_reward)
            
            # Track wins/losses
            if episode_reward > 0:
                wins += 1
            elif episode_reward < 0:
                losses += 1
            
            # End episode for agent (decay epsilon)
            agent.end_episode()
            
            # Progress report
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards[-100:])
                win_rate = wins / (episode + 1) * 100
                print(f"Episode {episode + 1}/{episodes} | "
                      f"Avg Reward: {avg_reward:.3f} | "
                      f"Win Rate: {win_rate:.1f}% | "
                      f"Epsilon: {agent.eps:.4f} | "
                      f"Q-table size: {len(agent.Q)}")
            
            # Save checkpoint
            if (episode + 1) % save_interval == 0:
                agent.save(f"{self.models_dir}/agent_rl_vs_ai_ep{episode+1}.pkl")
        
        game.close()
        
        # Save final model
        agent.save(f"{self.models_dir}/agent_rl_vs_ai_final.pkl")
        
        # Plot results
        plot_agent_reward(rewards, 
                         title="RL vs AI - Training Progress",
                         save_path=f"{self.graphe_dir}/rewards_rl_vs_ai.png")
        
        self.all_rewards['rl_vs_ai'] = rewards
        
        print(f"\nTraining complete!")
        print(f"Final Win Rate: {wins/episodes*100:.1f}%")
        print(agent)
        
        return agent, rewards


    def train_rl_vs_human(self, episodes=100, render=True):
        """
        Train/evaluate Q-Learning agent against human player.
        
        Mode: Agent RL vs Human
        The RL agent controls the LEFT paddle (opponent position)
        Human controls the RIGHT paddle with arrow keys
        
        Parameters
        ----------
        episodes : int
            Number of episodes to play
        render : bool
            Must be True for human interaction
            
        Returns
        -------
        tuple
            (agent, rewards_history)
        """
        print("=" * 60)
        print("MODE: Agent RL vs Human")
        print("=" * 60)
        print("Controls: UP/DOWN arrow keys")
        print("You control the RIGHT paddle (green)")
        print("The RL agent controls the LEFT paddle (red)")
        print("=" * 60)
        
        agent = ag.Qlearning(self.alpha, self.gamma, self.epsilon, self.eps_decay)
        
        # Try to load pre-trained model
        pretrained_path = f"{self.models_dir}/agent_rl_vs_ai_final.pkl"
        if os.path.exists(pretrained_path):
            agent.load(pretrained_path)
            print("Loaded pre-trained agent!")
        
        game = g.Game(agent, mode='rl_vs_human', render=True)
        
        rewards = []
        
        for episode in range(episodes):
            # In rl_vs_human mode, we need different state representation
            state = game.getStateKey2()  # State from opponent's perspective
            episode_reward = 0
            done = False
            steps = 0
            max_steps = 10000
            
            print(f"\nEpisode {episode + 1}/{episodes}")
            
            while not done and steps < max_steps:
                # Get action for RL agent (controlling opponent paddle)
                action = agent.get_action(state)
                
                # Execute action (human action is handled via keyboard events)
                new_state_player, reward, done = game.step(action)
                new_state = game.getStateKey2()
                
                # Invert reward for opponent agent
                agent_reward = -reward
                
                # Update Q-table
                new_action = agent.get_action(new_state)
                agent.update(state, new_state, action, new_action, agent_reward)
                
                episode_reward += reward
                state = new_state
                steps += 1
            
            rewards.append(episode_reward)
            agent.end_episode()
            
            if episode_reward > 0:
                print(f"You won! Reward: {episode_reward}")
            else:
                print(f"Agent won! Reward: {episode_reward}")
        
        game.close()
        
        # Save model
        agent.save(f"{self.models_dir}/agent_rl_vs_human.pkl")
        
        # Plot results
        plot_agent_reward(rewards,
                         title="RL vs Human - Game Results",
                         save_path=f"{self.graphe_dir}/rewards_rl_vs_human.png")
        
        self.all_rewards['rl_vs_human'] = rewards
        
        return agent, rewards


    def train_rl_vs_rl(self, episodes=1000, render=False, save_interval=100):
        """
        Train two Q-Learning agents against each other.
        
        Mode: Agent RL vs Agent RL
        
        Parameters
        ----------
        episodes : int
            Number of training episodes
        render : bool
            Whether to render the game
        save_interval : int
            Save Q-tables every N episodes
            
        Returns
        -------
        tuple
            (agent1, agent2, rewards_history)
        """
        print("=" * 60)
        print("MODE: Agent RL vs Agent RL")
        print("=" * 60)
        
        # Create two agents with slightly different parameters for diversity
        agent1 = ag.Qlearning(self.alpha, self.gamma, self.epsilon, self.eps_decay)
        agent2 = ag.Qlearning(self.alpha * 0.9, self.gamma, self.epsilon * 1.1, self.eps_decay)
        
        game = g.Game(agent1, agent2=agent2, mode='rl_vs_rl', render=render)
        
        rewards_agent1 = []
        rewards_agent2 = []
        
        agent1_wins = 0
        agent2_wins = 0
        
        for episode in range(episodes):
            state1 = game.reset()
            state2 = game.getStateKey2()
            
            episode_reward1 = 0
            episode_reward2 = 0
            done = False
            steps = 0
            max_steps = 5000
            
            while not done and steps < max_steps:
                # Get actions from both agents
                action1 = agent1.get_action(state1)
                action2 = agent2.get_action(state2)
                
                # Execute actions
                new_state1, reward, done = game.step(action1, action2)
                new_state2 = game.getStateKey2()
                
                # Rewards are inverted for the two agents
                reward1 = reward
                reward2 = -reward
                
                # Update both Q-tables
                new_action1 = agent1.get_action(new_state1)
                new_action2 = agent2.get_action(new_state2)
                
                agent1.update(state1, new_state1, action1, new_action1, reward1)
                agent2.update(state2, new_state2, action2, new_action2, reward2)
                
                episode_reward1 += reward1
                episode_reward2 += reward2
                
                state1 = new_state1
                state2 = new_state2
                steps += 1
            
            rewards_agent1.append(episode_reward1)
            rewards_agent2.append(episode_reward2)
            
            # Track wins
            if episode_reward1 > 0:
                agent1_wins += 1
            elif episode_reward1 < 0:
                agent2_wins += 1
            
            agent1.end_episode()
            agent2.end_episode()
            
            # Progress report
            if (episode + 1) % 100 == 0:
                avg_reward1 = np.mean(rewards_agent1[-100:])
                avg_reward2 = np.mean(rewards_agent2[-100:])
                print(f"Episode {episode + 1}/{episodes} | "
                      f"Agent1 Avg: {avg_reward1:.3f} | "
                      f"Agent2 Avg: {avg_reward2:.3f} | "
                      f"Agent1 Wins: {agent1_wins} | "
                      f"Agent2 Wins: {agent2_wins}")
            
            # Save checkpoints
            if (episode + 1) % save_interval == 0:
                agent1.save(f"{self.models_dir}/agent1_rl_vs_rl_ep{episode+1}.pkl")
                agent2.save(f"{self.models_dir}/agent2_rl_vs_rl_ep{episode+1}.pkl")
        
        game.close()
        
        # Save final models
        agent1.save(f"{self.models_dir}/agent1_rl_vs_rl_final.pkl")
        agent2.save(f"{self.models_dir}/agent2_rl_vs_rl_final.pkl")
        
        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(np.cumsum(rewards_agent1), 'b-', label='Agent 1 (Player)', linewidth=1)
        axes[0].plot(np.cumsum(rewards_agent2), 'r-', label='Agent 2 (Opponent)', linewidth=1)
        axes[0].set_title('Cumulative Rewards')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Cumulative Reward')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Win rate over time
        window = 100
        if len(rewards_agent1) >= window:
            win_rate1 = [np.mean([1 if r > 0 else 0 for r in rewards_agent1[max(0,i-window):i+1]]) 
                        for i in range(len(rewards_agent1))]
            axes[1].plot(win_rate1, 'b-', label='Agent 1 Win Rate', linewidth=1)
            axes[1].axhline(y=0.5, color='gray', linestyle='--', label='50%')
        axes[1].set_title('Agent 1 Win Rate (Rolling)')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Win Rate')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('RL vs RL - Training Progress', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{self.graphe_dir}/rewards_rl_vs_rl.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        self.all_rewards['rl_vs_rl_agent1'] = rewards_agent1
        self.all_rewards['rl_vs_rl_agent2'] = rewards_agent2
        
        print(f"\nTraining complete!")
        print(f"Agent 1 Win Rate: {agent1_wins/episodes*100:.1f}%")
        print(f"Agent 2 Win Rate: {agent2_wins/episodes*100:.1f}%")
        print(f"\nAgent 1:\n{agent1}")
        print(f"\nAgent 2:\n{agent2}")
        
        return agent1, agent2, rewards_agent1


    def demo_trained_agent(self, mode='rl_vs_ai', episodes=10):
        """
        Demo a trained agent with rendering enabled.
        
        Parameters
        ----------
        mode : str
            Game mode to demo
        episodes : int
            Number of episodes to demo
        """
        print(f"\nDemoing trained agent in {mode} mode...")
        
        agent = ag.Qlearning(self.alpha, self.gamma, 0.01)  # Low epsilon for exploitation
        
        # Load trained model
        model_path = f"{self.models_dir}/agent_{mode}_final.pkl"
        if os.path.exists(model_path):
            agent.load(model_path)
        else:
            print(f"No trained model found at {model_path}")
            print("Training a new agent first...")
            return
        
        game = g.Game(agent, mode=mode, render=True)
        
        for episode in range(episodes):
            state = game.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = agent.get_action(state)
                new_state, reward, done = game.step(action)
                episode_reward += reward
                state = new_state
                time.sleep(0.01)  # Slow down for visibility
            
            result = "Won!" if episode_reward > 0 else "Lost!"
            print(f"Episode {episode + 1}: {result} (Reward: {episode_reward})")
        
        game.close()


def main():
    """Main function to run all training modes."""
    print("=" * 60)
   
    print("=" * 60)
    
    # Initialize trainer
    trainer = GameLearning(
        alpha=0.5,      # Learning rate
        gamma=0.95,     # Discount factor
        epsilon=0.3,    # Initial exploration rate
        eps_decay=0.001 # Epsilon decay
    )
    
    # Menu
    print("\nSelect training mode:")
    print("1. Train RL vs AI (recommended first)")
    print("2. Play RL vs Human")
    print("3. Train RL vs RL")
    print("4. Demo trained agent")
    print("5. Run all modes")
    print("0. Exit")
    
    choice = input("\nEnter choice (0-5): ").strip()
    
    if choice == '1':
        render = input("Render game? (y/n): ").strip().lower() == 'y'
        episodes = int(input("Number of episodes (default 1000): ").strip() or "1000")
        trainer.train_rl_vs_ai(episodes=episodes, render=render)
        
    elif choice == '2':
        episodes = int(input("Number of episodes (default 10): ").strip() or "10")
        trainer.train_rl_vs_human(episodes=episodes)
        
    elif choice == '3':
        render = input("Render game? (y/n): ").strip().lower() == 'y'
        episodes = int(input("Number of episodes (default 1000): ").strip() or "1000")
        trainer.train_rl_vs_rl(episodes=episodes, render=render)
        
    elif choice == '4':
        trainer.demo_trained_agent(mode='rl_vs_ai', episodes=5)
        
    elif choice == '5':
        print("\n--- Training RL vs AI ---")
        trainer.train_rl_vs_ai(episodes=500, render=False)
        
        print("\n--- Training RL vs RL ---")
        trainer.train_rl_vs_rl(episodes=500, render=False)
        
        # Plot comparison
        plot_training_comparison(
            {'RL vs AI': trainer.all_rewards['rl_vs_ai'],
             'RL vs RL (Agent1)': trainer.all_rewards['rl_vs_rl_agent1']},
            save_path=f"{trainer.graphe_dir}/comparison_all_modes.png"
        )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
