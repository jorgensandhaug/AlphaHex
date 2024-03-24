import jax.numpy as jnp
from plants.plant import Plant

class CournotSystem(Plant):
    def __init__(self, p_max, c_m):
        self.p_max = p_max
        self.c_m = c_m

    def price(self, q):
        return self.p_max - q

    def profit(self, q1, q2):
        return (self.price(q1+q2) - self.c_m) * q1

    def max_possible_profit(self, q2):
        q1_max = (self.p_max - self.c_m - q2) / 2
        return q1_max, self.profit(q1_max, q2)

    def update(self, control_signal, disturbance, plant_state):
        plant_state["q1"] = plant_state["q1"] + control_signal
        plant_state["q2"] = plant_state["q2"] + disturbance
    
        # clip q1 and q2 to be between 0 and 1
        plant_state["q1"] = jnp.clip(plant_state["q1"], a_min=0, a_max=1)
        plant_state["q2"] = jnp.clip(plant_state["q2"], a_min=0, a_max=1)

        plant_state["output"] = self.profit(plant_state["q1"], plant_state["q2"])
        return plant_state