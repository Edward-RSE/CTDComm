from gymnasium import register

import predator_prey.env

register(
    id='PredatorPrey-v1',
    entry_point='predator_prey.env.predator_prey:env',
)