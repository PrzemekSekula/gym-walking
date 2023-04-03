import sys
from os import path
import numpy as np

import gym

from gym.error import DependencyNotInstalled

from typing import Optional

LEFT, RIGHT = 0, 1

class WalkingEnv(gym.Env):
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }
     
    def __init__(self, n_states = 7, render_mode = "human"):
        """Initializes environment

        Args:
            n_states (int, optional): Number of states (including 
                terminal states). Defaults to 7.
        """
        
        self.info = {'non terminal states': n_states}
        self.n_states = n_states  
        self.render_mode = render_mode
        
        self.state = int(np.floor(self.n_states / 2))
        
        
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(self.n_states)
        
        # pygame utils
        self.cell_size = (100, 100)
        self.window_size = (
            self.n_states * self.cell_size[0],
            2 * self.cell_size[0],
        )
        
        self.window_surface = None
        self.clock = None
        self.character_img = None
        self.state_bg_img = None
        self.terminal_bg_img = None
        self.character_bg_img = None
        self.text_font = None
        
    def step(self, action):
        if action == LEFT:
            self.state = self.state - 1
        elif action == RIGHT:
            self.state = self.state + 1
        else:
            raise ValueError("Invalid action")
        
     
        reward = 0
        terminal = False
        truncated = False
        if self.state == 0:
            terminal = True
        elif self.state == self.n_states - 1:
            terminal = True
            reward = 1
        
        return self.state, reward, terminal, truncated, self.info
    
    def reset(self, 
              *, 
              seed: Optional[int] = None, 
              options: Optional[dict] = None):
        """
        Resets the environment to an initial state and returns the 
        initial observation.
        
        Returns:
            state (object): the initial observation
            info (dict): a dictionary containing additional information
        """
        
        super().reset(seed=seed)
        
        self.state = int(np.floor(self.n_states / 2))
        
        return self.state, self.info
    
    
    def render(self, state_values = None):
        return self._render_gui(self.render_mode, state_values)
    
    
    def _render_gui(self, mode, state_values = None):
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[toy_text]`"
            )
        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("WalkingEnv")
                self.window_surface = pygame.display.set_mode(self.window_size)
            else:  # rgb_array
                self.window_surface = pygame.Surface(self.window_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()
            
            
        if self.text_font is None:
            self.text_font = pygame.font.SysFont("Arial", 25)            
            
        if self.character_img is None:
            file_name = path.join(path.dirname(__file__), "img/robot.png")
            self.character_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
                        
        if self.state_bg_img is None:
            file_name = path.join(path.dirname(__file__), "img/state_bg.png")
            self.state_bg_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        if self.terminal_bg_img is None:
            file_name = path.join(path.dirname(__file__), "img/terminal_bg.png")
            self.terminal_bg_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
            
        if self.character_bg_img is None:
            file_name = path.join(path.dirname(__file__), "img/black_bg.png")
            self.character_bg_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )   

        for row in range(self.n_states):
            
            col = 0
            pos = (row * self.cell_size[1], col * self.cell_size[0])
            
            self.window_surface.blit(self.character_bg_img, pos)
            if row == self.state:
                player_pos = (pos[0], pos[1])
                self.window_surface.blit(self.character_img, player_pos)
                
            col = 1
            pos = (row * self.cell_size[1], col * self.cell_size[0])
            if (row == 0) or (row == self.n_states - 1):
                self.window_surface.blit(self.terminal_bg_img, pos)
            else:
                self.window_surface.blit(self.state_bg_img, pos)
                if state_values is not None:
                    if len(state_values) == self.n_states - 2:
                        state_values = np.array([0] + list(state_values) + [0])
                    else:
                        state_values = np.array(state_values)
                        
                    if len(state_values) != self.n_states:
                        raise ValueError(f"state_values has {len_state_values} elements, {self.n_states} was expected.")

                    text = str(np.round(state_values[row], 3))

                    text_color = (255, 255, 255)
                    img = self.text_font.render(text, True, text_color)
                    self.window_surface.blit(
                        img, (
                        int(pos[0] + self.cell_size[0] / 2 - img.get_width() / 2),
                        int(pos[1] + self.cell_size[1] / 2 - img.get_height() / 2)
                        )
                    )
    

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )
       

