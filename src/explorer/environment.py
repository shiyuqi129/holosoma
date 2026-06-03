# Data structure and generation for simulation environment

from typing import Tuple, List
from enum import Enum
import random

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

def Prim(size: Tuple[int, int], cell_size: Tuple[int, int], wall_thickness: int = 1, r: float = 0):
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
    return grid

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
        self.grid = generation_method(size, *args, **kwargs)