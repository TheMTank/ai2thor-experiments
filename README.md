# cups-rl -- Customizable Unity Photorealistic Simulation for Reinforcement Learning algorithms

This project will focus primarily on the implementation and benchmark of different approaches to 
domain and task transfer learning in reinforcement learning. The focus lies on a diverse set of 
simplified domestic robot tasks using [ai2thor](https://ai2thor.allenai.org/), a realistic household 
3D environment. To provide an example, an agent could learn to pick up a cup under particular 
conditions and then zero/few shot transfer to pick up many different cups in many different 
situations.

We included our own wrapper for the environment as well to support the modification of the tasks 
within an openAI gym interface, so that new and more complex tasks can be developed efficiently to 
train and test the agent.

We will release a blog soon which will go into more detail about this repo and how to use all of the features of our wrapper.
More detailed information on ai2thor environment can be found on their 
[tutorial](http://ai2thor.allenai.org/tutorials/installation).


<div align="center">
  <img src="docs/bowls_fp_404_compressed_gif.gif" width="294px" />
  <p>A3C agent training on NaturalLanguagePickUpMultipleObjectTask in one of our customized scenes and tasks with the target object being CUPS!</p>
</div>

## Running algorithms on ai2thor

This project will include implementations and adaptations of the following papers as a benchmark of 
the current state of the art approaches to the problem:

- [A3C](https://arxiv.org/abs/1602.01783) [Code from Ikostrikov](https://github.com/ikostrikov/pytorch-a3c)
- [Gated-Attention Architectures for Task-Oriented Language Grounding](https://arxiv.org/abs/1706.07230) 
-- A3C with gated attention (A3C_LSTM_GA) *Original code available on [DeepRL-Grounding](https://github.com/devendrachaplot/DeepRL-Grounding)* 
also based on A3C made by Ikostrikov.

Implementations of these can be found in the algorithms folder and a3c can be run on AI2ThorEnv with:  
- `python algorithms/a3c/main.py`
- For running a config file which is set to the BowlsVsCups variant of the NaturalLanguagePickUpObjectTask in tasks.py for running A3C_LSTM_GA model:  
`python algorithms/a3c/main.py --config-file-name NL_pickup_bowls_vs_cups_fp1_config.json --verbose-num-steps True --num-random-actions-at-init 4`
- For running [ViZDoom](https://github.com/mwydmuch/ViZDoom) (you will need to install ViZDoom) synchronous with 1 process:  
`python algorithms/a3c/main.py --verbose-num-steps True --sync --vizdoom -v 1`
- For running atari with 8 processes:  
`python algorithms/a3c/main.py --atari --num-processes 8`

For A3C's `-eid` param you can specify experiment names which will create folders for checkpointing and hyperparameters, otherwise experiment name is the current date and a concatenated random guid. Check the argparse help for more details and variations of running the algorithm with different 
hyperparams. 

## Installation

Clone cups-rl repository:

```
# CUPS_RL=/path/to/clone/cups-rl
git clone https://github.com/TheMTank/cups-rl.git $CUPS_RL
```

Install Python dependencies (Currently only supporting python 3.5+):

`pip install -r $CUPS_RL/requirements.txt`

Finally, add `CUPS_RL` to your PYTHONPATH environment variable and you are done.

## How to use

The wrapper is based on OpenAI gym interfaces as described in [gym documentation](https://gym.openai.com/docs/).
Here is a simple example with the default configuration, that will place the agent in a "Kitchen" 
for the task of picking up and putting down mugs. 

```
from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
N_EPISODES = 20
env = AI2ThorEnv()
max_episode_length = env.task.max_episode_length
for episode in range(N_EPISODES):
    state = env.reset()
    for step_num in range(max_episode_length):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        if done:
            break
```

### Environment and Task configurations

##### JSON config files and config_dict

The environment is typically defined by a JSON configuration file located on the `gym_ai2thor/config_files` 
folder. You can find a full example at `default_config.json` to see how to customize it. Here there is 
another one as well:

```
# gym_ai2thor/config_files/myconfig.json
{'pickup_put_interaction': True,
 'open_close_interaction': true,
 'pickup_objects': ['Mug', 'Apple', 'Book'],
 'acceptable_receptacles': ['CounterTop', 'TableTop', 'Sink'],
 'openable_objects': ['Microwave'],
 'scene_id': 'FloorPlan28',
 'gridSize': 0.1,
 'continuous_movement': true,
 'grayscale': True,
 'resolution': (300, 300),
 'task': {'task_name': 'PickUp',
          'target_object': 'Mug'}} 
 ```
 
For experimentation it is important to be able to make slight modifications of the environment 
 without having to create a new config file each time. The class `AI2ThorEnv` includes the keyword 
 argument `config_dict`, that allows to input a python dictionary **in addition to** the config file 
 that overrides the parameters described in the config. In summary, the full interface to the constructor:  
 
 `env = AI2ThorEnv(env = AI2ThorEnv(config_file=config_file_name, config_dict=config_dict))` 

##### Tasks and TaskFactory

The tasks are defined in `envs/tasks.py` and allow for particular configurations regarding the 
rewards given and termination conditions for an episode. You can use the tasks that we defined
there or create your own modifying the `TaskFactory` and adding it as a subclass of the `BaseTask`. 
Here an example of a new task definition:

```
# envs/tasks.py
class TaskFactory:
    ...
    elif task_name == 'MoveAheadTask':
        return MoveAheadTask(**config['task'])
    ...
```

```
# envs/tasks.py
class MoveAheadTask(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(kwargs)
        self.rewards = []

    def transition_reward(self, state):
        reward = 1 if state.metadata['lastAction'] == 'MoveAhead' else -1 
        self.rewards.append(reward)
        done = sum(rewards) > 100 or self.step_num > self.max_episode_length
        if done:
            self.rewards = []
        return reward, done

    def reset(self):
        self.step_num = 0
```

Some tasks allow you return extra state by filling in the get_extra_state() function (e.g. for returning a Natural Language instruction within the state). Again, check 
tasks.py for more details.

##### Examples and Task variants

We encourage you to explore the scripts on the `examples` folder to guide you on the wrapper
 functionalities and explore how to create more customized versions of ai2thor environments and 
 tasks. 
 
 And most importantly, config files and tasks can be combined together to form **Task variants** e.g. NaturalLanguagePickUpObjectTask but only allowing 
 cups and bowls to be picked up hence: `gym_ai2thor/config_files/NL_pickup_bowls_vs_cups_fp1_config.json`

Here is the desired result of an example task in which the goal of the agent is to place a cup in the 
sink.

<div align="center">
  <img src="docs/cup_into_sink.gif" width="294px" />
  <p>Example of task "place cup in sink"</p>
</div>


## The Team

[MTank](http://www.themtank.org/) is a non-partisan organisation that works solely to recognise the multifaceted 
nature of Artificial Intelligence research and to highlight key developments within all sectors affected by these 
advancements. Through the creation of unique resources, the combination of ideas and their provision to the public, 
this project hopes to encourage the dialogue which is beginning to take place globally. 

To produce value for the individual, for researchers, for institutions and for the world.

## License

This project is released under the [MIT license](https://github.com/TheMTank/cups-rl/master/LICENSE).
