from __future__ import annotations

# Explorer
# Given a trained holosoma model, run it in a simulation environment
# The environment containes obstacle
# The program will provide the model with velocity to guide it to explore the environment

from dataclasses import dataclass
from pathlib import Path
import dataclasses
import tyro
import trimesh


from holosoma.utils import terrain_utils
from holosoma.config_types.logger import LoggerConfig, WandbLoggerConfig
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_types.terrain import MeshType, TerrainManagerCfg, TerrainTermCfg
from holosoma.config_types.video import VideoConfig
import holosoma.config_values.run_sim as run_sim_defaults

from environment import MazeEnvironmentGrid
from planner import KnownMapPlanner
from simulation import run_simulation, _run_checkpoint_simulation


@dataclass(frozen=True)
class ExplorerConfig:
    run_sim: RunSimConfig = dataclasses.replace(RunSimConfig(), simulator=run_sim_defaults.isaacgym)
    model_path: str | None = None
    use_wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_name: str | None = None
    video_enabled: bool = False
    video_interval: int = 1
    video_width: int = 640
    video_height: int = 360
    headless_recording: bool = False
    log_dir: str = "logs"
    max_eval_steps: int | None = None
    device: str = "cuda"


def build_logger_config(
    logger_config: LoggerConfig,
    use_wandb: bool,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_name: str | None,
    video_enabled: bool,
    video_interval: int,
    video_width: int,
    video_height: int,
    headless_recording: bool,
    log_dir: str,
) -> LoggerConfig:
    video_cfg = VideoConfig(
        enabled=video_enabled,
        interval=video_interval,
        width=video_width,
        height=video_height,
        save_dir=log_dir,
    )

    if use_wandb:
        return WandbLoggerConfig(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            video=video_cfg,
            headless_recording=headless_recording,
            base_dir=log_dir,
        )

    if isinstance(logger_config, WandbLoggerConfig):
        return dataclasses.replace(
            logger_config,
            video=video_cfg if video_enabled else logger_config.video,
            headless_recording=headless_recording or logger_config.headless_recording,
            base_dir=log_dir,
        )

    return dataclasses.replace(
        logger_config,
        video=video_cfg,
        headless_recording=headless_recording,
        base_dir=log_dir,
    )

def main() -> None:
    args = tyro.cli(ExplorerConfig)

    maze_size = (100, 100)
    cell_size = (2, 2)
    horizontal_scale = 0.1
    vertical_scale = 3.0

    maze = MazeEnvironmentGrid(maze_size, cell_size)

    planner = KnownMapPlanner(maze, horizontal_scale, vertical_scale, args.device)

    height_field = maze.grid
    vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
        height_field, horizontal_scale, vertical_scale
    )
    mesh: trimesh.Trimesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    mesh_path = Path("custom_terrain.obj")
    mesh.export(mesh_path)

    robot_cfg = dataclasses.replace(
        args.run_sim.robot,
        init_state=dataclasses.replace(args.run_sim.robot.init_state),
    )

    terrain_cfg = TerrainManagerCfg(
        terrain_term=TerrainTermCfg(
            func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
            mesh_type=MeshType.LOAD_OBJ,
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            obj_file_path=str(mesh_path),
            num_rows=1,
            num_cols=1,
        )
    )

    run_sim_config = dataclasses.replace(
        args.run_sim,
        training=dataclasses.replace(
            args.run_sim.training,
            headless=args.headless_recording or args.run_sim.training.headless,
        ),
        device=args.device
    )

    config = dataclasses.replace(
        run_sim_config,
        robot=robot_cfg,
        terrain=terrain_cfg,
        logger=build_logger_config(
            args.run_sim.logger,
            args.use_wandb,
            args.wandb_project,
            args.wandb_entity,
            args.wandb_name,
            args.video_enabled,
            args.video_interval,
            args.video_width,
            args.video_height,
            args.headless_recording,
            args.log_dir,
        ),
    )

    if args.model_path is not None:
        _run_checkpoint_simulation(config, planner, args.model_path, max_eval_steps=args.max_eval_steps)
    else:
        planner = KnownMapPlanner(maze, horizontal_scale, vertical_scale)
        run_simulation(config, planner)

if __name__ == "__main__":
    main()