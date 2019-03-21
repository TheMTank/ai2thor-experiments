"""
Here we try different cameraY (to bring the agent to the ground), gridSize (the amount of movement),
 continuous_movement (not just 90 degree rotations and can move diagonally) and finally a specific unity
 build. For ours we placed many cups on the ground. Still picks random actions but shows how much
 we can vary the environment.
"""
import time
import argparse

from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv

parser = argparse.ArgumentParser(description='Provide build path')
parser.add_argument('--build-file-name', required=True,
                    help='Required Unity build path for custom scenes (e.g. cups on ground)'
                         'necessary build file with folder should be in gym_ai2thor/build_files.'
                         'Also needs build folder with specific name in same path with file')
args = parser.parse_args()


if __name__ == '__main__':
    config_dict = {
        'pickup_put_interaction': True,
        'open_close_interaction': False,  # disable opening/closing objects
        'openable_objects': [],  # disable opening objects another way
        'pickup_objects': [
            "Cup"
        ],
        'scene_id': 'FloorPlan1',  # let's try a different room
        'grayscale': True,
        'resolution': [128, 128],
        'cameraY': -0.85,
        'gridSize': 0.1,  # 0.01
        'continuous_movement': True,
        'build_file_name': args.build_path,
        'task': {
            'task_name': 'PickUpTask',
            'target_object': 'Cup'
        }
    }

    # Input config_dict to env which will overwrite a few values given in the default config_file.
    # Therefore, a few warnings will occur
    env = AI2ThorEnv(config_dict=config_dict)
    max_episode_length = env.task.max_episode_length
    N_EPISODES = 3
    for episode in range(N_EPISODES):
        start = time.time()
        state = env.reset()
        for step_num in range(max_episode_length):
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            if done:
                break

            if step_num + 1 > 0 and (step_num + 1) % 100 == 0:
                print('Episode: {}. Step: {}/{}. Time taken for 100: {:.3f}s'.format(episode + 1,
                                         (step_num + 1), max_episode_length, time.time() - start))
                start = time.time()
