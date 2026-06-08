# Command hooks

from planner import BaseExplorerPlanner
from simulation import extract_observation

from holosoma.utils.safe_torch_import import torch

from holosoma.envs.base_task.base_task import BaseTask
from holosoma.managers.command.base import CommandTermBase
from holosoma.config_types.command import CommandTermCfg

class ExplorerCommandHooks(CommandTermBase):
    def __init__(self, cfg: CommandTermCfg, env: BaseTask):
        if env.num_envs > 1:
            raise ValueError("Explorer only support one environment at a time")
        self.planner: BaseExplorerPlanner = cfg.params.get("planner") # type: ignore
        if self.planner is None:
            raise ValueError("ExplorerCommandHooks requires a planner parameter.")
        if not isinstance(self.planner, BaseExplorerPlanner):
            raise TypeError("The planner parameter to ExplorerCommandHooks must be of type BaseExplorerPlanner")
        self.env = env
        self.smoothing = 0.8 # Arbitrary choice

    def setup(self):
        self.commands = torch.zeros(self.env.num_envs, 3, dtype=torch.float32, device=self.env.device)
        self.new_commands = torch.zeros(self.env.num_envs, 3, dtype=torch.float32, device=self.env.device)
        self.planner.fps = self.env.simulator.simulator_config.sim.fps

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        self.commands = torch.zeros(self.env.num_envs, 3, dtype=torch.float32, device=self.env.device)
        self.new_commands = torch.zeros(self.env.num_envs, 3, dtype=torch.float32, device=self.env.device)
        self.planner.reset()

    def step(self):
        obs = extract_observation(self.env.simulator)
        commands = self.planner.plan_motion(obs)
        if commands is not None:
            self.new_commands = commands
        
        self.commands[0, :3] = self.commands[0, :3] * self.smoothing + self.new_commands * (1-self.smoothing)
        self.env.simulator.commands[0, :3]= self.commands[0, :3]