# Handle simulation terrain generation
# Using heightmaps

from typing import Tuple

from environment import EnvironmentGrid
from holosoma.config_types.terrain import MeshType, TerrainTermCfg
from holosoma.simulator.shared.terrain import Terrain
from holosoma.utils.terrain_utils import SubTerrain, convert_heightfield_to_trimesh

import numpy as np
import trimesh

def get_explorer_terrain_term_config(grid_shape: Tuple[int, int], horizontal_scale: float, vertical_scale: float) -> TerrainTermCfg:
    return TerrainTermCfg(
        terrain_width=horizontal_scale * grid_shape[0],  # Total length based on grid width (number of rows)
        terrain_length=horizontal_scale * grid_shape[1],    # Total width based on grid length (number of columns)
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
        mesh_type=MeshType.TRIMESH,
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        num_rows=1,
        num_cols=1,
        terrain_config = { # Unused
            "flat": 0.0,
        }
    )

class ExplorerTerrain(Terrain):
    def __init__(self, num_robot: int, num_rows: int, num_cols: int, wall_height: float, horizontal_scale: float, vertical_scale: float, generation_method, *args, **kwargs):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.generation_method = generation_method
        self.wall_height = wall_height
        # for generation
        self.args = args
        self.kwargs = kwargs

        terrain_cfg = get_explorer_terrain_term_config((num_rows, num_cols), horizontal_scale=horizontal_scale, vertical_scale=vertical_scale) 

        super().__init__(terrain_cfg, num_robot)

    def randomized_terrain(self) -> None: # override
        self._height_field_raw = self.generation_method((self.num_rows, self.num_cols), *self.args, **self.kwargs) * self.wall_height / self._vertical_scale


###
def generate_from_grid_and_scale(env_grid: EnvironmentGrid, horizontal_scale: float, vertical_scale: float) -> Terrain:
    
    grid = np.array(env_grid.grid, dtype=int)

    subterrain = SubTerrain(
        terrain_name="grid_terrain",
        width=grid.shape[0],
        length=grid.shape[1],
        vertical_scale=vertical_scale,
        horizontal_scale=horizontal_scale
    )
    subterrain.height_field_raw[:] = grid

    terrain_cfg = TerrainTermCfg(
        func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
        mesh_type=MeshType.TRIMESH,
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        terrain_width=horizontal_scale * grid.shape[0],  # Total length based on grid width (number of rows)
        terrain_length=horizontal_scale * grid.shape[1],    # Total width based on grid length (number of columns)
        num_rows=1,
        num_cols=1,
        horizontal_scale=horizontal_scale,  # Scale for each grid cell in length
        vertical_scale=vertical_scale,    # Scale for height variations
        terrain_config = {
            "flat": 0.0,
        }
    )

    terrain = Terrain(terrain_cfg, 1)
    terrain.add_terrain_to_map(subterrain, 0, 0)

    vertices, triangles = convert_heightfield_to_trimesh(
        terrain._height_field_raw, terrain._horizontal_scale, terrain._vertical_scale, terrain._slope_threshold
    )
    mesh: trimesh.Trimesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    mesh.vertices[..., :2] -= terrain._border_size
    terrain._mesh = mesh

    return terrain

