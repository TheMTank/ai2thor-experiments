"""
Example use case of ai2thor wrapper. It runs N_EPISODES episodes in the Environment picking
random actions.
"""
import time

import gym
from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv

N_EPISODES = 3


if __name__ == '__main__':
    env = AI2ThorEnv()
    max_episode_length = env.task.max_episode_length
    for episode in range(N_EPISODES):
        start = time.time()
        state = env.reset()
        for step_num in range(max_episode_length):
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            if done:
                break

            if step_num + 1 > 0 and (step_num + 1) % 100 == 0:
                print('Episode: {}. Step: {}/{}. Time taken: {:.3f}s'.format(episode + 1,
                                         (step_num + 1), max_episode_length, time.time() - start))
                start = time.time()
