from gym.envs.registration import register

register(
  id='MicroSwimmer-v0',
  entry_point='envs.swimmer:MicroSwimmerEnv'
)

register(
  id='MicroSwimmer-v1',
  entry_point='envs.swimmer:DeepMicroSwimmerEnv'
)