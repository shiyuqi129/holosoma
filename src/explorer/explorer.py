# Explorer
# Given a trained holosoma model, run it in a simulation environment
# The environment containes obstacle
# The program will provide the model with velocity to guide it to explore the environment

from environment import Prim
from terrain import ExplorerTerrain
import numpy as np






def main():
    # Create terrain

    terrain = ExplorerTerrain(
        num_robot=1,
        num_rows=1000,
        num_cols=1000,
        wall_height=3
        horizontal_scale=0.1
        vertical_scale=0.1
        generation_method=Prim
    )

    # create observation grid, initialize to -1 (unknown)
    observation_grid: np.ndarray = np.full((1000, 1000), -1, dtype=int)

    raise NotImplementedError("Main function not fully implemented")

if __name__ == "__main__":
    main()