"""
Tests related to the ai2thor environment wrapper.
"""
import time
import unittest
import warnings

from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv


class TestAI2ThorEnv(unittest.TestCase):
    """
    General environment generation tests
    """
    def test_environments_runs(self):
        """
        Checks to see if the environment still runs and nothing breaks. Useful for continuous
        deployment and keeping master stable. Also, we check how much time 10 steps takes within
        the environment. Final assert checks if max_episode_length is equal to the number of steps
        taken and no off-by-one errors.

        Prints the execution time at the end of the test for performance check.
        """
        num_steps = 10
        env = AI2ThorEnv()
        start = time.time()
        all_step_times = []
        env.reset()
        for step_num in range(num_steps):
            start_of_step = time.time()
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)

            time_for_step = time.time() - start_of_step
            print('Step: {}. env.task.step_num: {}. Time taken for step: {:.3f}'.
                  format(step_num, env.task.step_num, time_for_step))
            all_step_times.append(time_for_step)

            if done:
                break

        print('Time taken altogether: {}\nAverage time taken per step: {:.3f}'.format(
            time.time() - start, sum(all_step_times) / len(all_step_times)))

        self.assertTrue(len(all_step_times) == num_steps)
        env.close()

    def test_cup_task_and_interaction_actions(self):
        """
        Check if picking up and putting down cup works and agent receives reward of 2 for doing it
        twice. For putting the cup down, the agent places it in the microwave and then picks it up
        again. Also this implicitly checks there is no random initialisation and that the same
        actions in the same environment will achieve the same reward each time.
        """

        actions_to_look_at_cup = ['RotateRight', 'RotateRight', 'MoveAhead', 'MoveAhead',
            'RotateRight', 'MoveAhead', 'MoveAhead', 'RotateLeft', 'MoveAhead', 'MoveAhead',
            'MoveAhead', 'RotateLeft', 'LookDown', 'PickupObject', 'PutObject', 'LookUp',
            'MoveRight', 'OpenObject', 'PutObject', 'PickupObject', 'CloseObject']

        env = AI2ThorEnv(config_dict={'scene_id': 'FloorPlan28',
                                      'gridSize': 0.25,
                                      'acceptable_receptacles': [
                                        'Microwave'  # the used receptacle below
                                      ]})

        for episode in range(2):  # twice to make sure no random initialisation
            env.reset()
            rewards = []
            for action_str in actions_to_look_at_cup:
                action = env.action_names.index(action_str)
                state, reward, done, _ = env.step(action)
                rewards.append(reward)
                if done:
                    break
            self.assertTrue(sum(rewards) == 2)

        env.close()

    def test_variations_of_natural_language_tasks(self):
        """
        test_natural_language_task look at task and other exceptions that should be raised
        """

        # mug actions were copied from before so is longer but episode should end before picking up
        actions_to_look_at_mug = ['RotateRight', 'RotateRight', 'MoveAhead', 'MoveAhead',
                                  'RotateRight', 'MoveAhead', 'MoveAhead', 'RotateLeft',
                                  'MoveAhead',
                                  'MoveAhead',
                                  'MoveAhead', 'RotateLeft', 'LookDown', 'PickupObject',
                                  'PutObject',
                                  'LookUp',
                                  'MoveRight', 'OpenObject', 'PutObject', 'PickupObject',
                                  'CloseObject']

        actions_to_look_at_apple = ['RotateRight', 'RotateRight', 'MoveAhead', 'MoveAhead',
                                    'RotateRight', 'MoveAhead', 'MoveAhead', 'MoveAhead',
                                    'RotateLeft', 'MoveAhead', 'MoveAhead', 'MoveAhead',
                                    'MoveAhead', 'MoveAhead', 'MoveAhead', 'LookDown',
                                    'MoveAhead', 'PickupObject']

        actions_to_look_at_tomato = actions_to_look_at_apple[:] + ['RotateLeft', 'MoveAhead',
                                                                   'RotateRight', 'MoveAhead',
                                                                   'PickupObject']

        # bread is behind apple
        actions_to_look_at_bread = actions_to_look_at_tomato[:]

        with self.assertRaises(ValueError):
            # reset needs to always be called before step
            env = AI2ThorEnv()
            env.step(0)
        env.close()

        with self.assertRaises(ValueError):
            # 'Cup' object type doesn't exist so ValueError is raised
            config_dict = {'num_random_actions_at_init': 3,
                           'lookupdown_actions': True,
                           'open_close_interaction': True,
                           'pickup_put_interaction': True,
                           'gridSize': 0.01,
                           'task': {
                               'task_name': 'NaturalLanguageLookAtObjectTask',
                               'list_of_instructions': ['Cup']
                           }}
            env = AI2ThorEnv(config_dict=config_dict)
            env.reset()
            env.step(0)
        env.close()

        config_dicts = [
                        {'lookupdown_actions': True,
                       'open_close_interaction': True,
                       'pickup_put_interaction': True,
                       'gridSize': 0.25,
                       'task': {
                           'task_name': 'NaturalLanguageLookAtObjectTask',
                           'list_of_instructions': ['Apple', 'Mug', 'Tomato', 'Bread', 'Chair']
                       }},
                        {'lookupdown_actions': True,
                         'open_close_interaction': True,
                         'pickup_put_interaction': True,
                         'pickup_objects': ['Apple', 'Mug', 'Chair'],
                         'gridSize': 0.25,
                         'task': {
                             'task_name': 'NaturalLanguagePickUpObjectTask',
                             'list_of_instructions': ['Apple', 'Mug', 'Chair']
                         }}]
        for config_dict in config_dicts:
            env = AI2ThorEnv(config_dict=config_dict)
            env.seed(42)
            for episode in range(12):
                state = env.reset()
                rewards = []
                if state[1] == 'Mug': current_set_of_actions = actions_to_look_at_mug
                elif state[1] == 'Apple': current_set_of_actions = actions_to_look_at_apple
                elif state[1] == 'Tomato': current_set_of_actions = actions_to_look_at_tomato
                elif state[1] == 'Bread': current_set_of_actions = actions_to_look_at_bread
                else: current_set_of_actions = actions_to_look_at_mug  # no reward for Chair

                for idx, action_str in enumerate(current_set_of_actions):
                    action = env.action_names.index(action_str)
                    state, reward, terminal, _ = env.step(action)
                    self.assertTrue(len(state) == 2)

                    if reward > 0:
                        print('Looked at: {} and got reward: {}. Episode over'.format(state[1],
                                                                                      reward))
                    rewards.append(reward)
                    if terminal:
                        break

                sum_of_rewards = sum(rewards)
                print('Sum of rewards: {}'.format(sum_of_rewards))
                if state[1] == 'Mug':
                    self.assertTrue(sum_of_rewards == 1)
                elif state[1] == 'Apple':
                    self.assertTrue(sum_of_rewards == 1)
                elif state[1] == 'Bread':
                    self.assertTrue(sum_of_rewards == 1)
                elif state[1] == 'Tomato':
                    self.assertTrue(sum_of_rewards == 1)
                else:
                    if config_dict['task']['task_name'] == 'NaturalLanguagePickUpObjectTask':
                        # picked up wrong object
                        self.assertTrue(sum_of_rewards == -1)
                    else:
                        # looked at wrong object so reward is 0
                        self.assertTrue(sum_of_rewards == 0)
            env.close()

    def test_config_override(self):
        """
        Check if reading both a config file and a config dict at the same time works and that the
        correct warning occurs for overwriting. Afterwards, check if scene_id was correctly
        changed from overwriting
        """
        with warnings.catch_warnings(record=True) as warning_objs:
            env = AI2ThorEnv(config_dict={'scene_id': 'FloorPlan27'})
            # checking if correct warning appears (there could be multiple depending on user)
            self.assertTrue([w for w in warning_objs if
                             'Key: scene_id already in config file' in w.message.args[0]])

        self.assertTrue(env.scene_id == 'FloorPlan27')
        env.close()


if __name__ == '__main__':
    unittest.main()
