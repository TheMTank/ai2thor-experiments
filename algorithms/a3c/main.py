"""
ActorCritic adapted from: https://github.com/ikostrikov/pytorch-a3c/blob/master/main.py
A3C_LSTM_GA adapted from: https://github.com/devendrachaplot/DeepRL-Grounding/blob/master/models.py
The main file needed within a3c. Runs of the train and test functions from their respective files.
Example of use:
`cd algorithms/a3c`
`python main.py`

Runs A3C on our AI2ThorEnv wrapper with default params (4 processes). Optionally it can be
run on any atari environment as well using the --atari and --atari-env-name params.
"""

from __future__ import print_function
import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"  # fixes multiprocessing error on some systems
import uuid
import glob
import json
import datetime

import torch
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
from algorithms.a3c.env_atari import create_atari_env
from algorithms.a3c import my_optim
from algorithms.a3c.model import ActorCritic, A3C_LSTM_GA
from algorithms.a3c.test import test
from algorithms.a3c.train import train


# Based on https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training/testing settings below
parser = argparse.ArgumentParser(description='A3C/A3C_GA')
# A3C settings
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for returns (default: 0.99)')
# todo rename tau to lambda and find out why it is higher than gamma?
'''
https://arxiv.org/pdf/1506.02438.pdf
. On the other hand, λ < 1
introduces bias only when the value function is inaccurate. Empirically, we find that the best value
of λ is much lower than the best value of 
'''
parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('--num-processes', type=int, default=4,
                    help='how many training processes to use (default: 4) except if synchronous')
parser.add_argument('-sync', '--synchronous', dest='synchronous', action='store_true',
                    help='Useful for debugging purposes e.g. import pdb; pdb.set_trace(). '
                         'Overwrites args.num_processes as everything is in main thread. '
                         '1 train() function is run and no test()')
parser.add_argument('-async', '--asynchronous', dest='synchronous', action='store_false')
parser.set_defaults(synchronous=False)

# experiment, environment and logging setting
parser.add_argument('-eid', '--experiment-id', default=False,
                    help='random or chosen guid for folder creation for plots and checkpointing.'
                         ' If experiment folder and id taken, will resume training!')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--test-sleep-time', type=int, default=200,
                    help='number of seconds to wait before testing again (default: 200)')
parser.add_argument('--checkpoint-freq', type=int, default=100000,
                    help='number of episodes passed for resuming')
parser.add_argument('--verbose-num-steps', default=False,
                    help='print step number every args.num_steps')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')

# ai2thor settings
parser.add_argument('--config-file-name', default='default_config.json',
                    help='File must be in gym_ai2thor/config_files. Other good variants with '
                         'natural language are: NL_pickup_5_objects_surround_fp_211_v0.1.json and'
                         'NL_pickup_multiple_cups_only_fp404_v0.1.json')
parser.add_argument('--resume-latest-config', default=1,
                    help='Whether to resume latest_config.json found in experiment folder'
                         'Default is set so user cannot override settings from text file')

parser.add_argument('--max-episode-length', type=int, default=1000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--num-random-actions-at-init', type=int, default=0,
                    help='Number of random actions the agent does on initialisation')

# Atari arguments. Good example of keeping code modular and allowing algorithms to run everywhere
parser.add_argument('--atari', dest='atari', action='store_true',
                    help='Run atari env instead with name below instead of ai2thor')
parser.add_argument('--atari-render', dest='atari_render', action='store_true',
                    help='Render atari')
parser.add_argument('--atari-env-name', default='PongDeterministic-v4',
                    help='environment to train on (default: PongDeterministic-v4)')
parser.set_defaults(atari=False)
parser.set_defaults(atari_render=False)

# VizDoom arguments
parser.add_argument('-d', '--difficulty', type=str, default="hard",
                    help="""Difficulty of the environment,
                    "easy", "medium" or "hard" (default: hard)""")
parser.add_argument('--living-reward', type=float, default=0,
                    help="""Default reward at each time step (default: 0,
                    change to -0.005 to encourage shorter paths)""")
parser.add_argument('--frame-width', type=int, default=300,
                    help='Frame width (default: 300)')
parser.add_argument('--frame-height', type=int, default=168,
                    help='Frame height (default: 168)')
parser.add_argument('-v', '--visualize', type=int, default=0,
                    help="""Visualize the envrionment (default: 0,
                    use 0 for faster training)""")
parser.add_argument('--sleep', type=float, default=0,
                    help="""Sleep between frames for better
                    visualization (default: 0)""")
parser.add_argument('--vizdoom', dest='vizdoom', action='store_true',
                    help='Run vizdoom env instead with name below instead of ai2thor and atari')
parser.add_argument('--scenario-path', type=str, default="vizdoom_maps/room.wad",
                    help="""Doom scenario file to load
                    (default: maps/room.wad)""")
parser.add_argument('--interactive', type=int, default=0,
                    help="""Interactive mode enables human to play
                    (default: 0)""")
parser.add_argument('--all-instr-file', type=str,
                    default="vizdoom_data/instructions_all.json",
                    help="""All instructions file path relative to a3c folder
                    (default: vizdoom_data/instructions_all.json)""")
parser.add_argument('--train-instr-file', type=str,
                    default="vizdoom_data/instructions_train.json",
                    help="""Train instructions file path relative to a3c folder
                    (default: vizdoom_data/instructions_train.json)""")
parser.add_argument('--test-instr-file', type=str,
                    default="vizdoom_data/instructions_test.json",
                    help="""Test instructions file path relative to a3c folder
                    (default: vizdoom_data/instructions_test.json)""")
parser.add_argument('--object-size-file', type=str,
                    default="vizdoom_data/object_sizes.txt",
                    help='Object size file path relative to a3c folder '
                         '(default: data/object_sizes.txt)')
parser.add_argument('-e', '--evaluate', type=int, default=0,
                    help="""0:Train, 1:Evaluate MultiTask Generalization
                    2:Evaluate Zero-shot Generalization (default: 0)
                    async must be on and will only run test function""")

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "" # todo add GA3C link for more GPU oriented version

    # todo add to README.md more info about all these eid settings.
    # todo Test it works from multiple folders
    # todo move experiments folder within a3c
    # todo resolve static method vs outside function issue
    # todo atari unbiased mean
    # todo synchronous comments fix
    # todo default atari not working and change to --game/env then string and add else
    # todo more docstrings on checkpoint loading
    # todo fix comments with fernando's suggestions below (checkpointing, shared_model,
    # todo save " ".join(sys.argv[:]) as well since im nervous about finding/writing command after resuming

    args = parser.parse_args()
    # set to 0 so that checkpoint resume can overwrite if necessary
    args.episode_number = 0
    args.total_length = 0
    args.num_backprops = 0
    if not args.experiment_id:
        args.experiment_id = datetime.datetime.now().strftime("%Y-%m-%d-") \
                                                                     + str(uuid.uuid4())
    args.experiment_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__),
                           '..', '..', 'experiments', args.experiment_id)))
    args.checkpoint_path = os.path.join(args.experiment_path, 'checkpoints')
    args.tensorboard_path = os.path.join(args.experiment_path, 'tensorboard_logs')

    # Create environment from choices of Atari, ViZDoom and our ai2thor wrapper with tasks
    if args.atari:
        env = create_atari_env(args.atari_env_name)
        args.resolution = (42, 42)  # fixed to be 42x42 for _process_frame42() in envs.py
    elif args.vizdoom:
        # many more dependencies required for VizDoom so therefore import here
        from algorithms.a3c.env_vizdoom import GroundingEnv

        if args.evaluate == 0:
            args.use_train_instructions = 1
        elif args.evaluate == 1:
            args.use_train_instructions = 1
            args.num_processes = 0
        elif args.evaluate == 2:
            args.use_train_instructions = 0
            args.num_processes = 0
        else:
            assert False, "Invalid evaluation type"

        env = GroundingEnv(args)
        env.game_init()
        args.resolution = (args.frame_width, args.frame_height)
    else:
        """ if you resume a checkpoint and if args.resume_latest_config == True, changes to the file
            will be overwritten by the latest_config.json file. Set to False if you want to change
            config settings in the config file after resuming and in the middle of training """
        args.last_config_resume_path = os.path.join(args.experiment_path, 'latest_config.json')
        args.config_dict = {}
        if args.resume_latest_config:
            print('args.resume_latest_config is True. Resuming latest config for the env+task')
            # read last_config.json if it exists, else create it below
            if os.path.exists(args.last_config_resume_path):
                print('Folder for this args.experiment_id "{}" and latest_config file existed'
                      .format(args.experiment_id))
                with open(args.last_config_resume_path, 'r') as f:
                    print('Loading config_dict from file: {}'.format(args.last_config_resume_path))
                    print('Therefore changes to args.config_file_name won\'t affect the env.\n'
                          'Set to args.resume_latest_config to 0 if you don\'t want this')
                    args.config_dict = json.load(f)
        else:
            print('Loading most params from args.config-file-name')

        # random actions on reset to encourage robustness
        args.config_dict['num_random_actions_at_init'] = args.num_random_actions_at_init
        args.config_dict['max_episode_length'] = args.max_episode_length
        # use given config file to start config and then allow config_dict to overwrite values
        config_file_dir_path = os.path.abspath(os.path.join(__file__, '../../..', 'gym_ai2thor',
                                                            'config_files'))
        args.config_file_path = os.path.join(config_file_dir_path, args.config_file_name)
        env = AI2ThorEnv(config_file=args.config_file_path, config_dict=args.config_dict,
                         seed=args.seed)  # todo decide
        args.resolution = (env.config['resolution'][0], env.config['resolution'][1])

    # Create shared model
    if env.task.task_has_language_instructions:
        # environment will return natural language sentence as part of state so process it with
        # Gated Attention (GA) variant of A3C
        shared_model = A3C_LSTM_GA(env.observation_space.shape[0], env.action_space.n,
                                   args.resolution, len(env.task.word_to_idx),
                                   args.max_episode_length)
    else:
        shared_model = ActorCritic(env.observation_space.shape[0], env.action_space.n,
                                   args.resolution)
    shared_model.share_memory()

    # Create optimizer
    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    env.close()  # above env initialisation was only to find certain params needed for models

    # Checkpoint creation/loading below
    checkpoint_counter = False
    if not os.path.exists(args.checkpoint_path):
        print('Tensorboard created experiment folder: {} and checkpoint folder '
              'made here: {}'.format(args.experiment_path, args.checkpoint_path))
        os.makedirs(args.checkpoint_path)
    else:
        print('Checkpoints path already exists at path: {}'.format(args.checkpoint_path))
        # look for checkpoints and find one with largest total_length
        checkpoint_paths = glob.glob(os.path.join(args.checkpoint_path, 'checkpoint_total_length*'))
        if checkpoint_paths:
            # Take checkpoint path with most experience
            # e.g. 2000 from checkpoint_total_length_2000.pth.tar
            checkpoint_file_name_ints = [
                int(x.split('/')[-1].split('.pth.tar')[0].split('_')[-1])
                for x in checkpoint_paths]
            # todo just get best? Stop saving each one?
            idx_of_latest = checkpoint_file_name_ints.index(max(checkpoint_file_name_ints))
            checkpoint_to_load = checkpoint_paths[idx_of_latest]
            print('Attempting to load latest checkpoint: {}'.format(checkpoint_to_load))

            # load checkpoint and unpack values
            if os.path.isfile(checkpoint_to_load):  # todo not needed
                print("Successfully loaded checkpoint {}".format(checkpoint_to_load))
                checkpoint = torch.load(checkpoint_to_load)
                args.total_length = checkpoint['total_length']
                args.episode_number = checkpoint['episode_number']
                args.num_backprops = checkpoint.get('num_backprops', 0)
                checkpoint_counter = checkpoint.get('counter', False)  # if not set, set to 0 later

                print('Values from checkpoint: total_length: {}. episode_number: {}'.format(
                    checkpoint['total_length'], checkpoint['episode_number']))
                shared_model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])

                print("=> loaded checkpoint '{}' (total_length {})"
                      .format(checkpoint_to_load, checkpoint['total_length']))
        else:
            print('No model checkpoint to load')

    # Creating TensorBoard writer and necessary folders
    writer = SummaryWriter(comment='A3C',  # this will create dirs
                           log_dir=args.tensorboard_path, purge_step=args.episode_number)
    # run tensorboardX --logs_dir args.tensorboard_path in terminal and open browser e.g.
    print('-----------------\nTensorboard command:\n'
          'tensorboard --logdir experiments/{}/tensorboard_logs'
          '\n-----------------'.format(args.experiment_id))

    # Save argparse arguments and environment config from last resume or first start
    with open(os.path.join(args.experiment_path, 'latest_args.json'), 'w') as f:
        args_dict = vars(args)
        args_dict['experiment_id'] = str(args.experiment_id)
        json.dump(args_dict, f)
    if not args.vizdoom and not args.atari:
        # Save ai2thor config
        with open(args.last_config_resume_path, 'w') as f:
            json.dump(env.config, f)

    # todo try fix multiprocessing again
    # process initialisation and training/testing starting
    processes = []
    counter = mp.Value('i', 0 if not checkpoint_counter else checkpoint_counter)
    lock = mp.Lock()
    try:
        if not args.synchronous:
            # test runs continuously and if episode ends, sleeps for args.test_sleep_time seconds
            p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter))
            p.start()
            processes.append(p)

            for rank in range(0, args.num_processes):
                p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock,
                                                   writer, optimizer))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        else:
            # synchronous so only 1 process. Best for debugging or just A2C
            rank = 0
            args.num_processes = 1
            # test(args.num_processes, args, shared_model, counter)  # check test functionality
            train(rank, args, shared_model, counter, lock, writer, optimizer)
    finally:
        writer.export_scalars_to_json(os.path.join(args.experiment_path, 'all_scalars.json'))
        writer.close()
