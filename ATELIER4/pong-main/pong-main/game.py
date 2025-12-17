"""
Pong game implementation with support for:

- Agent RL vs Agent AI
- Agent RL vs Human
- Agent RL vs Agent RL
"""
import pygame
import sys
import random

class Game:
    """
    Pong Game class with Q-Learning agent integration.
    
    Supports three game modes:
    - 'rl_vs_ai': Q-Learning agent (player) vs AI opponent
    - 'rl_vs_human': Q-Learning agent (opponent) vs Human player
    - 'rl_vs_rl': Q-Learning agent vs Q-Learning agent
    
    Parameters
    ----------
    agent : Qlearning
        The Q-learning agent controlling the player paddle
    agent2 : Qlearning, optional
        Second Q-learning agent for rl_vs_rl mode
    mode : str
        Game mode: 'rl_vs_ai', 'rl_vs_human', or 'rl_vs_rl'
    render : bool
        Whether to render the game visually
    """

    def __init__(self, agent, agent2=None, mode='rl_vs_ai', render=True):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.agent = agent          # Primary agent (controls player paddle)
        self.agent2 = agent2        # Secondary agent for rl_vs_rl mode
        self.mode = mode
        self.render_game = render
        
        # Screen settings
        self.screen_width = 1280
        self.screen_height = 820
        
        if self.render_game:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption(f"Pong - {mode.upper()}")
        else:
            # Headless mode for faster training
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
        
        # Game objects
        self.ball = pygame.Rect(self.screen_width / 2 - 15, self.screen_height / 2 - 15, 30, 30)
        self.player = pygame.Rect(self.screen_width - 20, self.screen_height / 2 - 70, 10, 140)
        self.opponent = pygame.Rect(10, self.screen_height / 2 - 70, 10, 140)
        
        # Colors
        self.bg_color = pygame.Color("grey12")
        self.light_grey = (200, 200, 200)
        self.player_color = (100, 200, 100)     # Green for RL agent
        self.opponent_color = (200, 100, 100)   # Red for opponent
        self.ball_color = (200, 200, 200)
        
        # Ball speed
        self.ball_speed_x = 7 * random.choice((1, -1))
        self.ball_speed_y = 7 * random.choice((1, -1))
        self.initial_ball_speed = 7
        
        # Paddle speeds
        self.player_speed = 0
        self.opponent_speed = 5  # Reduced from 7 to make AI beatable
        self.paddle_speed = 10  # Increased for agent advantage
        
        # AI difficulty settings
        self.ai_reaction_distance = 400  # AI only reacts when ball is close
        self.ai_error_rate = 0.15  # 15% chance AI makes a mistake
        
        # Scores
        self.player_score = 0
        self.opponent_score = 0
        
        # Font
        try:
            self.game_font = pygame.font.Font("freesansbold.ttf", 32)
        except:
            self.game_font = pygame.font.Font(None, 32)  # Use default font
        
        # Game state
        self.running = True
        self.episode_done = False
        
        # Reward tracking
        self.last_reward = 0

    def getStateKey(self):
        """
        Get the current state representation for the Q-learning agent.
        
        State is discretized to reduce state space:
        - Ball position (x, y) discretized into grid cells
        - Ball velocity direction (x, y)
        - Player paddle position (y) discretized
        - Relative position of ball to paddle
        
        Returns
        -------
        str
            String representation of the current state
        """
        # Discretize positions to reduce state space
        ball_x = int(self.ball.centerx // 80)  # 16 horizontal zones
        ball_y = int(self.ball.centery // 82)  # 10 vertical zones
        
        # Ball velocity direction
        ball_vx = 1 if self.ball_speed_x > 0 else -1
        ball_vy = 1 if self.ball_speed_y > 0 else 0 if self.ball_speed_y == 0 else -1
        
        # Player paddle position
        player_y = int(self.player.centery // 82)  # 10 vertical zones
        
        # Relative position of ball to paddle (for better learning)
        rel_y = 1 if self.ball.centery > self.player.centery else -1 if self.ball.centery < self.player.centery else 0
        
        state = f"{ball_x}_{ball_y}_{ball_vx}_{ball_vy}_{player_y}_{rel_y}"
        return state
    
    def getStateKey2(self):
        """
        Get state for the second agent (opponent side).
        
        Returns
        -------
        str
            String representation of the current state from opponent's perspective
        """
        ball_x = int(self.ball.centerx // 80)
        ball_y = int(self.ball.centery // 82)
        
        ball_vx = 1 if self.ball_speed_x > 0 else -1
        ball_vy = 1 if self.ball_speed_y > 0 else 0 if self.ball_speed_y == 0 else -1
        
        opponent_y = int(self.opponent.centery // 82)
        
        rel_y = 1 if self.ball.centery > self.opponent.centery else -1 if self.ball.centery < self.opponent.centery else 0
        
        state = f"{ball_x}_{ball_y}_{ball_vx}_{ball_vy}_{opponent_y}_{rel_y}"
        return state

    def move(self, action):
        """
        Move the player paddle based on action.
        
        Parameters
        ----------
        action : int
            -1 for up, 0 for stay, 1 for down
        """
        self.player.y += action * self.paddle_speed
        
        # Keep paddle within screen bounds
        if self.player.top <= 0:
            self.player.top = 0
        if self.player.bottom >= self.screen_height:
            self.player.bottom = self.screen_height
    
    def move_opponent(self, action):
        """
        Move the opponent paddle based on action (for rl_vs_rl mode).
        
        Parameters
        ----------
        action : int
            -1 for up, 0 for stay, 1 for down
        """
        self.opponent.y += action * self.paddle_speed
        
        if self.opponent.top <= 0:
            self.opponent.top = 0
        if self.opponent.bottom >= self.screen_height:
            self.opponent.bottom = self.screen_height

    def ball_animation(self):
        """
        Update ball position and handle collisions.
        
        Returns
        -------
        float
            Reward for the current step
        """
        reward = 0
        
        # Move ball
        self.ball.x += self.ball_speed_x
        self.ball.y += self.ball_speed_y
        
        # Top/bottom wall collision
        if self.ball.top <= 0 or self.ball.bottom >= self.screen_height:
            self.ball_speed_y *= -1
        
        # Left/right wall collision (scoring)
        if self.ball.left <= 0:
            # Player scores (opponent missed)
            self.player_score += 1
            reward = 1  # +1 if the player (RL agent) wins the point
            self.episode_done = True
            self.ball_restart()
            
        if self.ball.right >= self.screen_width:
            # Opponent scores (player missed)
            self.opponent_score += 1
            reward = -1  # -1 if the opponent (AI) wins the point
            self.episode_done = True
            self.ball_restart()
        
        # Paddle collisions
        if self.ball.colliderect(self.player):
            self.ball_speed_x *= -1
            self.ball.left = self.player.left - self.ball.width
            reward = 0.5  # +0.5 if the player touches the ball (increased reward)
            # Add some angle variation based on where ball hits paddle
            offset = (self.ball.centery - self.player.centery) / (self.player.height / 2)
            self.ball_speed_y = self.initial_ball_speed * offset * 0.8 + self.ball_speed_y * 0.2
            
        if self.ball.colliderect(self.opponent):
            self.ball_speed_x *= -1
            self.ball.right = self.opponent.right + self.ball.width
        
        self.last_reward = reward
        return reward

    def player_animation(self):
        """Update player paddle position (for human control mode)."""
        self.player.y += self.player_speed
        if self.player.top <= 0:
            self.player.top = 0
        if self.player.bottom >= self.screen_height:
            self.player.bottom = self.screen_height

    def opponent_ai(self):
        """AI opponent with limited reaction - beatable by RL agent."""
        # Only react when ball is coming towards opponent and within reaction distance
        if self.ball_speed_x < 0 and self.ball.centerx < self.ai_reaction_distance:
            # Add random error to make AI beatable
            if random.random() > self.ai_error_rate:
                if self.opponent.centery < self.ball.centery - 20:
                    self.opponent.y += self.opponent_speed
                elif self.opponent.centery > self.ball.centery + 20:
                    self.opponent.y -= self.opponent_speed
        else:
            # Move towards center when ball is far
            if self.opponent.centery < self.screen_height / 2 - 30:
                self.opponent.y += self.opponent_speed * 0.5
            elif self.opponent.centery > self.screen_height / 2 + 30:
                self.opponent.y -= self.opponent_speed * 0.5
        
        if self.opponent.top <= 0:
            self.opponent.top = 0
        if self.opponent.bottom >= self.screen_height:
            self.opponent.bottom = self.screen_height

    def ball_restart(self):
        """Reset ball to center with random direction."""
        self.ball.center = (self.screen_width / 2, self.screen_height / 2)
        self.ball_speed_x = self.initial_ball_speed * random.choice((1, -1))
        self.ball_speed_y = self.initial_ball_speed * random.choice((1, -1))

    def reset(self):
        """Reset game state for new episode."""
        self.ball.center = (self.screen_width / 2, self.screen_height / 2)
        self.player.centery = self.screen_height / 2
        self.opponent.centery = self.screen_height / 2
        self.ball_speed_x = self.initial_ball_speed * random.choice((1, -1))
        self.ball_speed_y = self.initial_ball_speed * random.choice((1, -1))
        self.player_speed = 0
        self.episode_done = False
        self.last_reward = 0
        return self.getStateKey()

    def step(self, action, action2=None):
        """
        Execute one game step.
        
        Parameters
        ----------
        action : int
            Action for the player agent
        action2 : int, optional
            Action for the second agent (in rl_vs_rl mode)
            
        Returns
        -------
        tuple
            (new_state, reward, done) for player agent
        """
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            # Human controls (for rl_vs_human mode)
            if self.mode == 'rl_vs_human':
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_DOWN:
                        self.player_speed = 8
                    if event.key == pygame.K_UP:
                        self.player_speed = -8
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_DOWN or event.key == pygame.K_UP:
                        self.player_speed = 0
        
        # Move paddles based on mode
        if self.mode == 'rl_vs_ai':
            # RL agent controls player, AI controls opponent
            self.move(action)
            self.opponent_ai()
        elif self.mode == 'rl_vs_human':
            # RL agent controls opponent, human controls player
            self.move_opponent(action)
            self.player_animation()
        elif self.mode == 'rl_vs_rl':
            # Both paddles controlled by RL agents
            self.move(action)
            if action2 is not None:
                self.move_opponent(action2)
        
        # Update ball and get reward
        reward = self.ball_animation()
        
        # Render if enabled
        if self.render_game:
            self.render()
        
        new_state = self.getStateKey()
        return new_state, reward, self.episode_done

    def render(self):
        """Render the game."""
        self.screen.fill(self.bg_color)
        
        # Draw center line
        pygame.draw.aaline(self.screen, self.light_grey, 
                          (self.screen_width / 2, 0), 
                          (self.screen_width / 2, self.screen_height))
        
        # Draw paddles with different colors
        pygame.draw.rect(self.screen, self.player_color, self.player)
        pygame.draw.rect(self.screen, self.opponent_color, self.opponent)
        
        # Draw ball
        pygame.draw.ellipse(self.screen, self.ball_color, self.ball)
        
        # Draw scores
        player_txt = self.game_font.render(f"Player: {self.player_score}", False, self.light_grey)
        opponent_txt = self.game_font.render(f"Opponent: {self.opponent_score}", False, self.light_grey)
        self.screen.blit(player_txt, (self.screen_width - 200, 20))
        self.screen.blit(opponent_txt, (20, 20))
        
        # Draw mode info
        mode_txt = self.game_font.render(f"Mode: {self.mode}", False, self.light_grey)
        self.screen.blit(mode_txt, (self.screen_width / 2 - 100, 20))
        
        pygame.display.flip()
        self.clock.tick(60)

    def play(self):
        """
        Main game loop for interactive play (original method).
        """
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_DOWN:
                        self.player_speed += 6
                    if event.key == pygame.K_UP:
                        self.player_speed -= 6
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_DOWN:
                        self.player_speed -= 6
                    if event.key == pygame.K_UP:
                        self.player_speed -= 6
            self.ball_animation()
            self.player_animation()
            self.opponent_ai()
            if self.render_game:
                self.render()

    def close(self):
        """Close the game window."""
        pygame.quit()
