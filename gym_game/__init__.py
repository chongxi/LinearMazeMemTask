from gym.envs.registration import register

register(
    id='Pygame-v0',
    entry_point='gym_game.envs:RaceEnv',
    max_episode_steps=2000,
)

register(
    id='MemTask-v0',
    entry_point='gym_game.envs:MemEnv',
    max_episode_steps=2000,
)
