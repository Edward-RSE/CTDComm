from gymnasium import register

register(
    id='PredatorPrey-v1',
    entry_point='decentralised-envs.dec_predator_prey_env:env',
)