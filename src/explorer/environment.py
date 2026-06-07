# Data structure and generation for simulation environment

from typing import Tuple, List
from enum import Enum
import random
import numpy as np

class _MazeCell:
    def __init__(self, walls: Tuple[int, int, int, int] = (1,1,1,1)):
        '''
        walls: 1=wall, 0=no wall. north, south, east, west, in order
        '''
        self.north, self.south, self.east, self.west = walls
        self.connected = False

class EnviromentGridCellType(Enum):
    EMPTY=0
    OCCUPIED=1
    UNKNOWN=-1

def print_grid(grid):
    c=['  ','**']
    for row in grid:
        for i in row:
            print(c[i], end='')
        print('')


def _draw_rectangle(grid, start_row, end_row, start_col, end_col, fill = 1, DoICareAboutIndexError = False):
    '''
    Literally just:
    for i in range(start_row, end_row):
        for j in range(start_col, end_col):
            try:
                grid[i][j] = fill
            except IndexError as e:
                if DoICareAboutIndexError:
                    raise e
    '''
    for i in range(start_row, end_row):
        for j in range(start_col, end_col):
            try:
                grid[i][j] = fill
            except IndexError as e:
                if DoICareAboutIndexError:
                    raise e

def Prim(size: Tuple[int, int], cell_size: Tuple[int, int], wall_thickness: int = 1, r: float = 0) -> np.ndarray:
    '''
    size: size of the grid
    cell_size: size of each cell
    wall_thickness: thickness of the walls
    r: rate of wall removal

    The perimeter will be wall
    '''
    if not 0 <= r <= 1:
        raise ValueError("Argument 'r' for 'Prim' should be between 0 and 1")
    n_row = (size[0] - 2 + wall_thickness) // (cell_size[0] + wall_thickness)
    n_col = (size[1] - 2 + wall_thickness) // (cell_size[1] + wall_thickness)
    if size[0] < cell_size[0]+2 or size[1] < cell_size[1]+2:
        raise ValueError(f"Grid size {size} is too small to fit cell size {cell_size} with walls {wall_thickness} wide and 1-wide walls on perimeter")

    maze_cells = [[_MazeCell() for col in range(n_col)] for row in range(n_row)]

    wall_list: List[Tuple[int, int, Tuple[int, int]]] = []

    row, col = random.randint(0, n_row-1), random.randint(0, n_col-1)
    maze_cells[row][col].connected = True
    wall_list.extend([(row, col, dir) for dir in [(1, 0), (-1, 0), (0, 1), (0, -1)]])

    def is_valid_coord(row: int, col: int) -> bool:
        return 0<=row<n_row and 0<=col<n_col

    def remove_wall(row: int, col: int, dir: Tuple[int, int]) -> None:
        if dir == (1, 0):
            maze_cells[row][col].south = 0
            maze_cells[row+1][col].north = 0
        elif dir == (-1, 0):
            maze_cells[row][col].north = 0
            maze_cells[row-1][col].south = 0
        elif dir == (0, 1):
            maze_cells[row][col].east = 0
            maze_cells[row][col+1].west = 0
        elif dir == (0, -1):
            maze_cells[row][col].west = 0
            maze_cells[row][col-1].east = 0
        else:
            raise ValueError(f"{dir} is not a valid direction")

    while wall_list:
        i = random.randrange(0, len(wall_list))
        row, col, dir = wall_list[i]
        wall_list[i] = wall_list[-1]
        wall_list.pop()
        neighbor_row = row+dir[0]
        neighbor_col = col+dir[1]
        if not is_valid_coord(neighbor_row, neighbor_col):
            continue
        if maze_cells[neighbor_row][neighbor_col].connected == True:
            if random.random() < r:
                remove_wall(row, col, dir)
        else:
            remove_wall(row, col, dir)
            maze_cells[neighbor_row][neighbor_col].connected = True
            wall_list.extend([(neighbor_row, neighbor_col, dir) for dir in [(1, 0), (-1, 0), (0, 1), (0, -1)]])
    
    grid = [[0 for col in range(size[1])] for row in range(size[0])]

    _draw_rectangle(grid, 0, 1, 0, size[1])
    _draw_rectangle(grid, 0, size[0], 0, 1)

    for row in range(n_row):
        for col in range(n_col):
            wall_start_row = row * (cell_size[0] + wall_thickness) - wall_thickness + 1
            wall_start_col = col * (cell_size[1] + wall_thickness) - wall_thickness + 1
            if row != 0 and maze_cells[row][col].north:
                _draw_rectangle(grid, wall_start_row, wall_start_row + wall_thickness, wall_start_col, wall_start_col + cell_size[1] + 2 * wall_thickness)
            if col != 0 and maze_cells[row][col].west:
                _draw_rectangle(grid, wall_start_row, wall_start_row + cell_size[0] + 2 * wall_thickness, wall_start_col, wall_start_col + wall_thickness)

    _draw_rectangle(grid, n_row * (cell_size[0] + wall_thickness) - wall_thickness + 1, size[0], 0, size[1])
    _draw_rectangle(grid, 0 , size[0], n_col * (cell_size[1] + wall_thickness) - wall_thickness + 1, size[1])
    return np.array(grid, dtype=float)

class EnvironmentGridGenerationMethod(Enum):
    PRIM = Prim

class EnvironmentGrid:
    def __init__(self, size: Tuple[int, int], generation_method = Prim, *args, **kwargs):
        '''
        size: size of the grid, must be the first argument of generation_method
        generation_method: method of generation, can use EnvironmentGridGenerationMethod enum or pass in your own function
        args, kwargs: arguments for method, check each method for reference
        '''
        self.size = size
        self.grid: np.ndarray = generation_method(size, *args, **kwargs)

class MazeEnvironmentGrid(EnvironmentGrid):
    def __init__(self, size: Tuple[int, int], cell_size: Tuple[int, int], wall_thickness: int = 1, r: float = 0, generation_method = Prim, **kwargs):
        super().__init__(size, generation_method, cell_size = cell_size, wall_thickness = wall_thickness, r = r, **kwargs)
        self.cell_size = cell_size
        self.wall_thickness = wall_thickness
        self.r = r

    @property
    def n_grid_row(self):
        return self.grid.shape[0]
    @property
    def n_grid_col(self):
        return self.grid.shape[1]
    @property
    def n_cell_row(self):
        return (self.n_grid_row - 2 + self.wall_thickness) // (self.cell_size[0] + self.wall_thickness)
    @property
    def n_cell_col(self):
        return (self.n_grid_col - 2 + self.wall_thickness) // (self.cell_size[1] + self.wall_thickness)
    
    def cell2coord(self, row: int, col: int, point = 'center')-> Tuple[int, int]:
        '''
        Convert cell index to grid coordinate
        point: which point of the cell to convert, can be 'center', 'top_left', 'top_right', 'bottom_left', 'bottom_right'
        '''
        grid_row = row * (self.cell_size[0] + self.wall_thickness) + 1
        grid_col = col * (self.cell_size[1] + self.wall_thickness) + 1
        if point == 'center':
            grid_row += self.cell_size[0] // 2
            grid_col += self.cell_size[1] // 2
        elif point == 'top_left':
            pass
        elif point == 'top_right':
            grid_col += self.cell_size[1] - 1
        elif point == 'bottom_left':
            grid_row += self.cell_size[0] - 1
        elif point == 'bottom_right':
            grid_row += self.cell_size[0] - 1
            grid_col += self.cell_size[1] - 1
        else:
            raise ValueError(f"{point} is not a valid point type")

        return grid_row, grid_col
    
    def coord2cell(self, grid_row: int, grid_col: int) -> Tuple[int, int]:
        '''
        Convert grid coordinate to cell index
        '''
        row = (grid_row - 1) // (self.cell_size[0] + self.wall_thickness)
        col = (grid_col - 1) // (self.cell_size[1] + self.wall_thickness)
        return row, col
    
    def is_wall(self, grid_row: int, grid_col: int, direction: Tuple[int, int]) -> bool:
        '''
        Check if there is a wall in the given direction from the given grid coordinate
        direction: (1, 0) for south, (-1, 0) for north, (0, 1) for east, (0, -1) for west
        '''
        cell_row, cell_col = self.coord2cell(grid_row, grid_col)
        return self.is_wall_cell_coord(cell_row, cell_col, direction)
        
    def is_wall_cell_coord(self, cell_row: int, cell_col: int, direction: Tuple[int, int]) -> bool:
        if not (0 <= cell_row < self.n_cell_row and 0 <= cell_col < self.n_cell_col):
            raise ValueError(f"Cell coordinate ({cell_row}, {cell_col}) is out of bounds")
        grid_coord = self.cell2coord(cell_row, cell_col, point='top_left')
        if direction == (1, 0):
            return self.grid[grid_coord[0] + self.cell_size[0], grid_coord[1]] == 1
        elif direction == (-1, 0):
            return self.grid[grid_coord[0] - 1, grid_coord[1]] == 1
        elif direction == (0, 1):
            return self.grid[grid_coord[0], grid_coord[1] + self.cell_size[1]] == 1
        elif direction == (0, -1):
            return self.grid[grid_coord[0], grid_coord[1] - 1] == 1
        else:
            raise ValueError(f"{direction} is not a valid direction")

    def path_find(self, start_cell_row, start_cell_col, end_cell_row, end_cell_col) -> List[Tuple[int, int]]:
        '''
        Find a path from start cell to end cell using BFS
        '''
        if not (0 <= start_cell_row < self.n_cell_row and 0 <= start_cell_col < self.n_cell_col):
            raise ValueError(f"Start cell coordinate ({start_cell_row}, {start_cell_col}) is out of bounds")
        if not (0 <= end_cell_row < self.n_cell_row and 0 <= end_cell_col < self.n_cell_col):
            raise ValueError(f"End cell coordinate ({end_cell_row}, {end_cell_col}) is out of bounds")

        from collections import deque

        queue = deque()
        queue.append((start_cell_row, start_cell_col))
        visited = set()
        visited.add((start_cell_row, start_cell_col))
        parent = dict()

        while queue:
            cell_row, cell_col = queue.popleft()
            if (cell_row, cell_col) == (end_cell_row, end_cell_col):
                break
            
            for dir in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if not self.is_wall_cell_coord(cell_row, cell_col, dir):
                    neighbor_row = cell_row + dir[0]
                    neighbor_col = cell_col + dir[1]
                    if (neighbor_row, neighbor_col) not in visited and 0 <= neighbor_row < self.n_row and 0 <= neighbor_col < self.n_col:
                        visited.add((neighbor_row, neighbor_col))
                        parent[(neighbor_row, neighbor_col)] = (cell_row, cell_col)
                        queue.append((neighbor_row, neighbor_col))

        path = []
        current = (end_cell_row, end_cell_col)
        while current != (start_cell_row, start_cell_col):
            path.append(current)
            current = parent[current]
        path.append((start_cell_row, start_cell_col))
        path.reverse()
        return path