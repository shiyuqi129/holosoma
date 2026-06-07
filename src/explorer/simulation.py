'''
Manage simulation
'''

from planner import BaseExplorerPlanner
from command_hooks import ExplorerCommandHooks

import dataclasses
import sys
import traceback
import time

#import tyro
from loguru import logger

from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_types.logger import LoggerConfig
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.command import CommandTermCfg, CommandManagerCfg

from holosoma.utils.sim_utils import DirectSimulation, setup_simulation_environment, close_simulation_app
from holosoma.utils.eval_utils import CheckpointConfig, load_checkpoint, load_saved_experiment_config
from holosoma.utils.helpers import get_class

def extract_observation(simulator):
    # Example observation from the simulator tensors:
    return {
        "dof_pos": simulator.dof_pos[0, :].clone(),
        "dof_vel": simulator.dof_vel[0, :].clone(),
        "base_state": simulator.robot_root_states[0, :7].clone(),
    }


def run_simulation(config: RunSimConfig, planner: BaseExplorerPlanner):
    """Run simulation with direct simulator control.

    This function provides direct access to the simulator for continuous simulation
    with bridge support using the DirectSimulation class.

    Parameters
    ----------
    config : RunSimConfig
        Configuration containing all simulation settings.
    """
    # Auto-set device for GPU-accelerated backends if still on default CPU
    if config.device == "cpu":
        # Check if using Warp backend (requires CUDA)
        if hasattr(config.simulator.config, "mujoco_backend"):
            from holosoma.config_types.simulator import MujocoBackend  # noqa: PLC0415 -- deferred

            if config.simulator.config.mujoco_backend == MujocoBackend.WARP:
                logger.info("Auto-detected MuJoCo Warp backend - setting device to cuda:0")
                config = dataclasses.replace(config, device="cuda:0")

    config = dataclasses.replace(config, device=config.device)

    logger.info("Starting Holosoma Direct Simulation...")
    logger.info(f"Robot: {config.robot.asset.robot_type}")
    logger.info(f"Simulator: {config.simulator._target_}")
    logger.info(f"Terrain: {config.terrain.terrain_term.mesh_type} ({config.terrain.terrain_term.func})")

    try:
        # Use shared utils for setup
        env, device, simulation_app = setup_simulation_environment(config, device=config.device)

        # Create and run direct simulation using context manager for automatic clean-up
        with DirectSimulation(config, env, device, simulation_app) as sim:
            sim.initialize()
            planner.fps = config.simulator.config.sim.fps

            while True:
                sim.simulator.refresh_sim_tensors()

                observation = extract_observation(sim.simulator)
                target_vel = planner.plan_motion(observation)

                # Write the commanded locomotion
                if target_vel is not None:
                    sim.simulator.commands[0, :3] = target_vel

                sim.simulator.simulate_at_each_physics_step()
                sim.simulator.render()

                time.sleep(1.0 / config.simulator.config.sim.fps)

    except Exception as e:
        logger.error(f"Error during simulation: {e}")
        traceback.print_exc()
        sys.exit(1)


def _build_experiment_config_for_checkpoint(
    planner: BaseExplorerPlanner,
    checkpoint: str,
    run_sim_config: RunSimConfig,
    logger_config: LoggerConfig,
    headless_recording: bool,
) -> tuple[ExperimentConfig, ExperimentConfig, str | None, str]:
    loaded_config, wandb_run_path = load_saved_experiment_config(CheckpointConfig(checkpoint=checkpoint))
    eval_config = loaded_config.get_eval_config()

    eval_training = dataclasses.replace(
        eval_config.training,
        headless=headless_recording or eval_config.training.headless,
        num_envs=1,
    )

    eval_config = dataclasses.replace(
        eval_config,
        terrain=run_sim_config.terrain,
        logger=logger_config,
        training=eval_training,
        command=CommandManagerCfg(
            setup_terms={"explorer" : CommandTermCfg(func = "command_hooks:ExplorerCommandHooks",
                                                     params = {"planner": planner})},
            reset_terms={"explorer" : CommandTermCfg(func = "command_hooks:ExplorerCommandHooks")},
            step_terms={"explorer" : CommandTermCfg(func = "command_hooks:ExplorerCommandHooks")},
        )
    )

    checkpoint_path = str(load_checkpoint(checkpoint, logger_config.base_dir))
    return loaded_config, eval_config, wandb_run_path, checkpoint_path


def _run_checkpoint_simulation(config: RunSimConfig, planner, checkpoint: str, max_eval_steps: int | None = None) -> None:
    loaded_config, eval_config, wandb_run_path, checkpoint_path = _build_experiment_config_for_checkpoint(
        planner,
        checkpoint,
        config,
        config.logger,
        config.training.headless,
    )

    env, device, simulation_app = setup_simulation_environment(eval_config, planner)

    try:
        algo_class = get_class(eval_config.algo._target_)
        algo = algo_class(
            device=device,
            env=env,
            config=eval_config.algo.config,
            log_dir=eval_config.logger.base_dir,
            multi_gpu_cfg=None,
        )
        algo.setup()
        algo.attach_checkpoint_metadata(loaded_config, wandb_run_path)
        algo.load(checkpoint_path)
        algo.evaluate_policy(max_eval_steps=max_eval_steps)
    finally:
        if simulation_app:
            close_simulation_app(simulation_app)