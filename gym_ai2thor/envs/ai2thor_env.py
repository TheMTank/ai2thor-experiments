"""
Base class implementation for ai2thor environments wrapper, which adds an openAI gym interface for
inheriting the predefined methods and can be extended for particular tasks.
"""
import random
import os
import warnings

import ai2thor.controller
import numpy as np
from skimage import transform
import gym
from gym import error, spaces
from gym.utils import seeding

from gym_ai2thor.image_processing import rgb2gray
from gym_ai2thor.utils import read_config
from gym_ai2thor.tasks import TaskFactory
from gym_ai2thor.task_utils import calculate_euc_distance_between_agent_and_object

ALL_POSSIBLE_ACTIONS = [
    'MoveAhead',
    'MoveBack',
    'MoveRight',
    'MoveLeft',
    'LookUp',
    'LookDown',
    'RotateRight',
    'RotateLeft',
    'OpenObject',
    'CloseObject',
    'PickupObject',
    'PutObject'
    # Rotate is also possible when continuous_movement == True but we don't list it here
    # Teleport and TeleportFull but these shouldn't be allowable actions for an agent
]


class AI2ThorEnv(gym.Env):
    """
    This class wraps the ai2thor environment
    """
    def __init__(self, config_file='config_files/default_config.json', config_dict=None):
        """
        :param config_file:  (str)   Path to environment configuration file. Either absolute or
                                     relative path to the root of this repository.
                                     default and good example at: config_files/config_example.json
        :param: config_dict: (dict)  Overrides specific fields from the input configuration file.
        """
        # Loads config settings from file
        print('Reading in base config file at: {}'.format(config_file))
        self.config = read_config(config_file, config_dict)
        # Randomness settings
        self.np_random = None
        self.num_random_actions_at_init = self.config.get('num_random_actions_at_init', 0)
        # Object settings
        # acceptable objects taken from config file.
        self.allowed_objects = {}
        if self.config['pickup_put_interaction']:
            self.allowed_objects['pickupables'] = self.config['pickup_objects']
            self.allowed_objects['receptacles'] = self.config.get('acceptable_receptacles', [])
            if not self.allowed_objects['receptacles']:
                warnings.warn('list of receptacles is empty or wasn\'t given')
        if self.config['open_close_interaction']:
            self.allowed_objects['openables'] = self.config['openable_objects']
        # Action settings
        self.action_names = tuple(ALL_POSSIBLE_ACTIONS.copy())
        # remove open/close and pickup/put actions if respective interaction bool is set to False
        if not self.config['open_close_interaction']:
            # Don't allow opening and closing if set to False
            self.action_names = tuple([action_name for action_name in self.action_names if 'Open'
                                       not in action_name and 'Close' not in action_name])
        if not self.config['pickup_put_interaction']:
            self.action_names = tuple([action_name for action_name in self.action_names if 'Pickup'
                                       not in action_name and 'Put' not in action_name])
        if self.config.get('put_interaction'):
            self.action_names = tuple([action_name for action_name in self.action_names
                                       if 'Put' not in action_name])
        if not self.config.get('lookupdown_actions', True):
            self.action_names = tuple([action_name for action_name in self.action_names
                                       if 'Look' not in action_name])
        if not self.config.get('moveupdown_actions', True):
            self.action_names = tuple([action_name for action_name in self.action_names
                                       if 'Move' not in action_name])
        self.action_space = spaces.Discrete(len(self.action_names))
        # rotation settings
        self.continuous_movement = self.config.get('continuous_movement', False)
        if self.continuous_movement:
            self.absolute_rotation = 0.0
            self.rotation_amount = 10.0

        # Image settings
        self.event = None
        channels = 1 if self.config['grayscale'] else 3
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(channels, self.config['resolution'][0],
                                                   self.config['resolution'][1]),
                                            dtype=np.uint8)
        # ai2thor initialise function settings
        self.cameraY = self.config.get('cameraY', 0.0)
        self.gridSize = self.config.get('gridSize', 0.1)
        
        # Create task from config
        self.task = TaskFactory.create_task(self.config)
        # set scene_id
        if self.task.random_scene_ids_on_reset:
            self.scene_id = random.choice(self.task.random_scene_ids_on_reset)
        else:
            self.scene_id = self.config.get('scene_id')
        if not self.scene_id:
            raise ValueError('Need to specify scene_id in config')
        # todo check for self.config['task'].get('list_of_xyz_starting_positions')
        # todo and implement random.choice of starting position for specific task/scene with teleport
        # todo precompute all valid locations for all scenes and avoid objects too?
        # Start ai2thor
        self.controller = ai2thor.controller.Controller()
        if self.config.get('build_file_name'):
            # file must be in gym_ai2thor/build_files
            self.build_file_path = os.path.abspath(os.path.join(__file__, '../../build_files',
                                                   self.config['build_file_name']))
            print('Build file path at: {}'.format(self.build_file_path))
            if not os.path.exists(self.build_file_path):
                raise ValueError('Unity build file at:\n{}\n does not exist'.format(
                    self.build_file_path))
            self.controller.local_executable_path = self.build_file_path
        self.controller.start()

        self.reset_ever = False

    def step(self, action, verbose=True):
        if not self.action_space.contains(action):
            raise error.InvalidAction('Action must be an integer between '
                                      '0 and {}!'.format(self.action_space.n))
        if not self.reset_ever:
            raise ValueError('Cannot step() in environment if it has not been reset() before')
        action_str = self.action_names[action]
        visible_objects = [obj for obj in self.event.metadata['objects'] if obj['visible']]

        # if/else statements below for dealing with up to 13 actions
        if action_str.endswith('Object'):  # All interactions end with 'Object'
            # Interaction actions
            interaction_obj, distance = None, float('inf')
            inventory_before = [x['objectType'] for x in
                                   self.event.metadata['inventoryObjects']] \
                                   if self.event.metadata['inventoryObjects'] else []
            if action_str.startswith('Put'):
                closest_receptacle = None
                for obj in visible_objects:
                    # look for closest receptacle to put object from inventory
                    if obj['receptacle'] and obj['distance'] < distance \
                        and obj['objectType'] in self.allowed_objects['receptacles'] \
                            and len(obj['receptacleObjectIds']) < obj['receptacleCount']:
                        closest_receptacle = obj
                        distance = closest_receptacle['distance']
                if self.event.metadata['inventoryObjects'] and closest_receptacle:
                    interaction_obj = closest_receptacle
                    object_to_put = self.event.metadata['inventoryObjects'][0]
                    self.event = self.controller.step(
                            dict(action=action_str,
                                 objectId=object_to_put['objectId'],
                                 receptacleObjectId=interaction_obj['objectId']))
                    self.event.metadata['lastObjectPut'] = object_to_put
                    self.event.metadata['lastObjectPutReceptacle'] = interaction_obj
            elif action_str.startswith('Pickup'):
                closest_pickupable = None
                for obj in visible_objects:
                    # look for closest object to pick up
                    if obj['pickupable'] and obj['distance'] < distance and \
                            obj['distance'] < self.task.max_object_pickup_crosshair_dist and \
                            obj['objectType'] in self.allowed_objects['pickupables']:
                        if self.task.max_object_pickup_euclidean_dist:
                            euc_distance_to_obj = calculate_euc_distance_between_agent_and_object(
                                self.event.metadata['agent'], obj)
                            # print('3D dist: {}. distance to crosshair: {}'.format(
                            #     euc_distance_to_obj, obj['distance'])) # todo remove
                            if euc_distance_to_obj < self.task.max_object_pickup_euclidean_dist:
                                closest_pickupable = obj
                        else:
                            closest_pickupable = obj
                if closest_pickupable: # and not self.event.metadata['inventoryObjects']: # todo option to check for this@?
                    interaction_obj = closest_pickupable
                    self.event = self.controller.step(
                        dict(action=action_str, objectId=interaction_obj['objectId']))
                    self.event.metadata['lastObjectPickedUp'] = interaction_obj
            elif action_str.startswith('Open'):
                closest_openable = None
                for obj in visible_objects:
                    # look for closest closed receptacle to open it
                    if obj['openable'] and obj['distance'] < distance and not obj['isopen'] and \
                            obj['objectType'] in self.allowed_objects['openables']:
                        closest_openable = obj
                        distance = closest_openable['distance']
                    if closest_openable:
                        interaction_obj = closest_openable
                        self.event = self.controller.step(
                            dict(action=action_str, objectId=interaction_obj['objectId']))
                        self.event.metadata['lastObjectOpened'] = interaction_obj
            elif action_str.startswith('Close'):
                closest_openable = None
                for obj in visible_objects:
                    # look for closest opened receptacle to close it
                    if obj['openable'] and obj['distance'] < distance and obj['isopen'] and \
                            obj['objectType'] in self.allowed_objects['openables']:
                        closest_openable = obj
                        distance = closest_openable['distance']
                    if closest_openable:
                        interaction_obj = closest_openable
                        self.event = self.controller.step(
                            dict(action=action_str, objectId=interaction_obj['objectId']))
                        self.event.metadata['lastObjectClosed'] = interaction_obj
            else:
                raise error.InvalidAction('Invalid interaction {}'.format(action_str))
            # print what object was interacted with and state of inventory
            if interaction_obj and verbose:
                inventory_after = [x['objectType'] for x in
                                   self.event.metadata['inventoryObjects']] \
                                   if self.event.metadata['inventoryObjects'] else []
                if action_str in ['PutObject', 'PickupObject']:
                    inventory_changed_str = 'Inventory before/after: {}/{}.'.format(
                                                            inventory_before, inventory_after)
                else:
                    inventory_changed_str = ''
                print('{}: {}. {}'.format(
                    action_str, interaction_obj['objectType'], inventory_changed_str))
        elif action_str.startswith('Rotate'):
            if self.continuous_movement:
                # Rotate actions
                if action_str.endswith('Left'):
                    self.absolute_rotation -= self.rotation_amount
                    self.event = self.controller.step(
                        dict(action='Rotate', rotation=self.absolute_rotation))
                elif action_str.endswith('Right'):
                    self.absolute_rotation += self.rotation_amount
                    self.event = self.controller.step(
                        dict(action='Rotate', rotation=self.absolute_rotation))
            else:
                # Do normal RotateLeft/Right command in discrete mode (i.e. 3D GridWorld)
                self.event = self.controller.step(dict(action=action_str))
        else:
            # Move and Look actions
            self.event = self.controller.step(dict(action=action_str))

        self.task.step_num += 1

        # todo if frame_segmentation_on: return segmented image. OR could add it as a channel?
        # todo create 50 random room tasks with segmentation on since it should be easier
        image_state = self.preprocess(self.event.frame)
        extra_state = self.task.get_extra_state()
        state = (image_state, extra_state) if extra_state else image_state

        reward, done = self.task.transition_reward(self.event)

        info = {}
        return state, reward, done, info

    def preprocess(self, img):
        """
        Compute image operations to generate state representation
        """
        img = transform.resize(img, self.config['resolution'], mode='reflect')
        img = img.astype(np.float32)
        if self.observation_space.shape[0] == 1:
            img = rgb2gray(img)  # todo cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.moveaxis(img, 2, 0)
        return img

    def reset(self):
        print('Resetting environment and starting new episode')
        self.reset_ever = True
        if self.task.random_scene_ids_on_reset:
            self.scene_id = random.choice(self.task.random_scene_ids_on_reset)
            print('Changing to scene_id: {}'.format(self.scene_id))
        self.controller.reset(self.scene_id)
        self.event = self.controller.step(dict(action='Initialize', gridSize=self.gridSize,
                                               cameraY=self.cameraY, renderDepthImage=True,
                                               renderClassImage=True, renderObjectImage=True,
                                               continuous=self.continuous_movement))
        self.task.step_num = 0  # needed so done returns False below. task.reset() also does this

        if self.num_random_actions_at_init > 0:
            for i in range(self.num_random_actions_at_init):
                action = self.action_space.sample()
                _, _, done, _ = self.step(action)
                if done:
                    return self.reset()  # if we end episode in random actions, reset again

        if self.continuous_movement:
            self.absolute_rotation = 0.0
        extra_state = self.task.reset()
        image_state = self.preprocess(self.event.frame)

        if extra_state:
            state = (image_state, extra_state)
        else:
            state = image_state

        return state

    def render(self, mode='human'):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        random.seed(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        return seed1

    def close(self):
        self.controller.stop()
