# Command hooks

from planner import BaseExplorerPlanner
from simulation import extract_observation

import torch

from holosoma.holosoma.envs.base_task.base_task import BaseTask
from holosoma.holosoma.managers.command.base import CommandTermBase

class ExplorerCommandHooks(CommandTermBase):
    def __init__(self, planner: BaseExplorerPlanner, env: BaseTask):
        if env.num_envs > 1:
            raise ValueError("Explorer only support one environment at a time")
        self.planner = planner
        self.env = env

    def setup(self):
        self.command = torch.zeros()
        self.planner.fps = self.env.simulator.simulator_config.sim.fps

    def reset(self, env_ids):
        self.planner.reset()

    def step(self):
        obs = extract_observation(self.env.simulator)
        commands = self.planner.plan_motion(obs)
        self.env.simulator.commands[0, :3]= commands