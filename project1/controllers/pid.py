from controllers.controller import Controller
import numpy as np
import jax
import jax.numpy as jnp


# PID Controller subclass
class PIDController(Controller):
    def __init__(self, limit_output=None):
        self.limit_output = limit_output

    def forward(self, params, error_history, i):
        error = error_history[i]
        derivative = jax.lax.cond(i == 0, lambda error_history, i: error_history[i], lambda error_history, i: error_history[i] - error_history[i-1], error_history, i)
        integral = jnp.sum(error_history)

        output = params[0] * error + params[1] * integral + params[2] * derivative

        if self.limit_output is not None:
            output = jnp.clip(output, a_min=-self.limit_output[0], a_max=self.limit_output[1])

        return output

    def update_parameters(self, params, grad, learning_rate):
        ps = jax.tree_map(lambda p, g: p - learning_rate * g, params, grad)
        # clip parameters to be positive
        ps = np.clip(ps, a_min=0, a_max=None)
        return ps