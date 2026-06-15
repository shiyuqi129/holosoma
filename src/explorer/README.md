# Explorer

The following is things you need to know before and when running the codes

## Feature
This project include on single feature, provide a robot instruction to help it navigate through a maze. The result and be recorded as video.

## Failure
The robot is currently being dropped from 3 meters in the air into the map and it is likely that it will fall to the ground immediately. If a simulation lasted longer than a few seconds, it is almost certain that the robot has survived the initial drop and should be good for the rest of the simulation.

## Running
### Train a Model
Please reference [holosoma](../holosoma/README.md) for how to train a model
### Run Explorer
Activate the environment `hsgym` you created in the last step.

You can run with the command

`python explorer.py --model-path path/to/your/model.pt`

To record a video, run
`python explorer.py --model-path path/to/your/model.pt --video-enabled`

or

`python explorer.py --model-path path/to/your/model.pt --headless-recording`

### About Video Recording
The default max length of a episode is 60 seconds, I haven't make it an option so you'll have to change it manually in line 128 of [`simulation.py`](simulation.py), for example, if you want it to be 2 minutes (120 seconds), you can change that line to say `max_episode_length_s=120`.

It is recommended that you set `max_eval_steps`, here is how you can choose a good value for your use case:
- Take the max length of a episode and multiply it by 50, that's about how many step required for a video of that length
- Add a bit extra steps (I recommend 500-1000, you can add even more if you want to be safe) to account for the robot failing to survive *the drop*.
- Finally, multiply by how many videos you want to get

After recording, you should find the videos in the log directory (default `log/`), you can find the useful videos, i.e., those where the robot didn't fall immediately, by finding the longer (or bigger) video files. (you wouldn't be able to do this is the max length is small, but why would you record such short videos and expect to see anything?)

### About the result
At the top of the screen, you will see some texts, which is the command given to the robot at that time. Only the `vx`, `vy`, and `ang_vel` is relevant, as the other two are unused and will always be 0.

The robot is walking toward a target. You can visually see when the robot reaches the target because I made the robot stop and stand in place for about 4 seconds when it does. So do not panic when you see the robot seemingly stopped for no reason, instead, you should be happy that the robot completed one task.

## Maze Options
You can change the maze's parameter, it is listed in line 94-100 of `explorer.py`

### Layout
`maze_size` is the size of the grid used to store the maze's layout, it is also used to generate mesh for the terrain so it is not recommended to make it too big.

`cell_size` is how big a maze cell is on the maze grid

`wall_thickness` is how wide the walls are in the grid, it is not recommended to set it to one as the mesh generated will have very thin walls

`r` is the chance that a wall between two already connected cells will be removed during the maze generating process. 0 will create a simple maze with no loop and 1 will create a empty maze

`wall_height` is the height of the wall in meters

### Scale:
`horizontal_scale` will affect `maze_size`, `cell_size`, and `wall_thickness`, multiplying them by the scale will be their value in meters

`vertical_scale` is mostly irrelevant as `wall_height` is already in meters, however, it might change how the resulting mesh get generated so it is recommended to keep it at some small values.

