from gym.envs.registration import register

register(id='ai2thor-v0',
         entry_point='gym_ai2thor.envs:AI2ThorEnv')
