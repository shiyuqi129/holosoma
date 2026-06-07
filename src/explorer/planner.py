# Motion Planning

from __future__ import annotations

from environment import MazeEnvironmentGrid

import numpy as np
import random

from holosoma.utils.safe_torch_import import torch

from typing import Tuple
from abc import ABC, abstractmethod

from holosoma.utils.rotations import calc_heading

class BaseExplorerPlanner(ABC):
    def __init__(self, device) -> None:
        self.fps: int | None = None
        self.device = device
        pass

    @abstractmethod
    def plan_motion(self, observation: dict) -> torch.Tensor | None:
        '''
        Given an observation, plan the next motion for the robot.
        
        Output:
            [vx, vy, wz] in m/s and rad/s
        '''
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
class ExplorerPlanner(BaseExplorerPlanner):
    def __init__(self, maze: MazeEnvironmentGrid, horizontal_scale: float, wall_height: float, device) -> None:
        super().__init__(device)
        self.maze = maze
        self.true_grid = maze.grid
        self.horizontal_scale = horizontal_scale
        self.wall_height = wall_height

        # Initializa observed grid with unknown (-1)
        self.observed_grid = np.full_like(maze.grid, -1, dtype=int)

    def plan_motion(self, observation: dict) -> torch.Tensor | None:
        raise NotImplementedError("motion planning not implemented")
    
    def get_current_grid_coord(self, pos: tuple[float, float]) -> Tuple[int, int]:
        '''
        Translate world position to grid coordinate
        pos: world_x and world_y
        '''
        return int(pos[0]//self.horizontal_scale), int(pos[1]//self.horizontal_scale)
    
    def get_current_cell_coord(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        grid_coord = self.get_current_grid_coord(pos)
        return self.maze.coord2cell(*grid_coord)
    
    def grid2world(self, grid_coord: Tuple[int, int]) -> Tuple[float, float]:
        return grid_coord[0] * self.horizontal_scale, grid_coord[1] * self.horizontal_scale
    
    def cell2world(self, cell_coord: Tuple[int, int]) -> Tuple[float, float]:
        grid_row, grid_col = self.maze.cell2coord(*cell_coord)
        return self.grid2world((grid_row, grid_col))
    
    @property
    def max_vel(self) -> float:
        return 1.0 # Arbitrary choice
    @property
    def max_ang_vel(self) -> float:
        return 1.0 # Arbitrary choice
    
    def reset(self):
        pass

    
class KnownMapPlanner(ExplorerPlanner):
    '''
    With full knowledge of the layout
    '''
    def __init__(self, maze: MazeEnvironmentGrid, horizontal_scale: float, wall_height: float, device) -> None:
        super().__init__(maze, horizontal_scale, wall_height, device)

        self.target_cell = None
        self.planned_path = None
        self.path_index = 0

        self.planner_cooldown = 0 # To not plan every step

    def plan_motion(self, observation: dict) -> torch.Tensor | None:
        '''
        Only support one robot
        Return none when no new command
        '''

        self.planner_cooldown -= 1
        if self.planner_cooldown > 0:
            return None

        current_pos: torch.Tensor = observation['base_state'][:2] # Assuming x, y are the first two elements
        current_x = current_pos[0].item()
        current_y = current_pos[1].item()

        current_row, current_col = self.get_current_cell_coord((current_x, current_y))

        if self.target_cell == None:
            # Pick a random target
            while True:
                row, col = random.randrange(0, self.maze.n_cell_row), random.randrange(0, self.maze.n_cell_col)
                if self.maze.grid[row][col] == 0 and (row, col) != (current_row, current_col):
                    self.target_cell = (row, col)
                    break
            # Plan the path to target

            self.planned_path = self.maze.path_find(current_row, current_col, self.target_cell[0], self.target_cell[1])

        if self.planned_path is None or self.path_index >= len(self.planned_path):
            # Retry
            self.target_cell = None
            self.planned_path = None
            self.path_index = 0
            self.planner_cooldown = 0
            return None
        
        if (current_row, current_col) == self.planned_path[self.path_index+1]:
            self.path_index+=1
            if self.path_index == len(self.planned_path)-1:
                self.target_cell = None
                self.planned_path = None
                self.path_index = 0
                self.planner_cooldown = self.fps if self.fps is not None else 20 # 1 second victory pause
                return torch.zeros(3)
            
        # Plan motion
        last_row, last_col = self.planned_path[self.path_index]
        next_row, next_col = self.planned_path[self.path_index+1]

        next_x, next_y = self.cell2world((next_row, next_col))

        path_vector_norm = torch.tensor([next_row-last_row, next_col-last_col], dtype=torch.float, device=self.device)
        target_vector = torch.tensor([next_x-current_x, next_y-current_y], dtype=torch.float, device=self.device)

        heading_angle = calc_heading(observation['base_state'][3:7].unsqueeze(0)).squeeze()
        heading_vector = torch.stack([torch.cos(heading_angle), torch.sin(heading_angle)])
        heading_vector_perp = torch.stack([-torch.sin(heading_angle), torch.cos(heading_angle)])

        along = (target_vector @ path_vector_norm) * path_vector_norm
        cross = target_vector - along

        desired_velocity = along + cross * (1 + torch.linalg.norm(cross))

        if torch.linalg.norm(desired_velocity) > self.max_vel:
            desired_velocity = torch.nn.functional.normalize(desired_velocity, dim=-1) * self.max_vel

        vx = desired_velocity @ heading_vector
        vy = desired_velocity @ heading_vector_perp

        angle_offset = torch.atan2(heading_vector[0]*desired_velocity[1]-heading_vector[1]*desired_velocity[0], heading_vector@desired_velocity)

        w = torch.clamp(5 * angle_offset, -self.max_ang_vel, self.max_ang_vel)

        return torch.stack([vx, vy, w])

    def reset(self):
        self.path_index = 0
        self.target_cell = None
        self.planned_path = None
        self.planner_cooldown = 0