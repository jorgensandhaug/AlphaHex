import jax.numpy as jnp
from plants.plant import Plant

class Bathtub(Plant):
    def __init__(self, A, C):
        self.A = A
        self.C = C

    def update(self, control_signal, disturbance, plant_state):
        H = plant_state["output"]
        
        sqrt_term = jnp.sqrt(2 * 9.81 * H) * self.C
        out = (H + control_signal + disturbance - sqrt_term) / self.A
        # clip H to be positive
        out = jnp.clip(out, a_min=0, a_max=None)
        plant_state["output"] = out
        return plant_state
