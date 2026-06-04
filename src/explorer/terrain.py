# Handle simulation terrain generation
# Using heightmaps

from environment import EnvironmentGrid
from holosoma.config_types.terrain import MeshType, TerrainTermCfg
from holosoma.simulator.shared.terrain import Terrain
from holosoma.utils.terrain_utils import SubTerrain, convert_heightfield_to_trimesh

import numpy as np
import trimesh


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

