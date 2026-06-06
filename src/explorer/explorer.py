# Explorer
# Given a trained holosoma model, run it in a simulation environment
# The environment containes obstacle
# The program will provide the model with velocity to guide it to explore the environment

from environment import MazeEnvironmentGrid
from planner import KnownMapPlanner

from pathlib import Path
import dataclasses
import trimesh

from holosoma.utils import terrain_utils
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_types.terrain import MeshType, TerrainManagerCfg, TerrainTermCfg
import holosoma.config_values.run_sim as run_sim_defaults
import holosoma.config_values.robot as robot_defaults
from simulation import run_simulation


def get_target_velocity() -> list[float]:
    # Replace with your own target angular velocity generator.
    return [0.0, 0.0, 1.2]  # [wx, wy, wz] in rad/s

def main() -> None:
    maze_size = (1000, 1000)
    cell_size = (10, 10)
    horizontal_scale = 0.1
    vertical_scale = 3.0


    maze = MazeEnvironmentGrid(maze_size, cell_size)

    height_field = maze.grid
    vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
        height_field, horizontal_scale, vertical_scale
    )
    mesh: trimesh.Trimesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    mesh_path = Path("custom_terrain.obj")
    mesh.export(mesh_path)

    robot_cfg = dataclasses.replace(
        robot_defaults.g1_29dof,
        init_state=dataclasses.replace(
            robot_defaults.g1_29dof.init_state,
        ),
    )

    terrain_cfg = TerrainManagerCfg(
        terrain_term=TerrainTermCfg(
            func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
            mesh_type=MeshType.LOAD_OBJ,
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            obj_file_path=str(mesh_path),
        )
    )

    config = RunSimConfig(
        simulator=run_sim_defaults.isaacgym,
        robot=robot_cfg,
        terrain=terrain_cfg,
        device="cuda:0",  # or "cpu"
    )

    planner = KnownMapPlanner(maze, horizontal_scale, vertical_scale)

    run_simulation(config, planner)

if __name__ == "__main__":
    main()