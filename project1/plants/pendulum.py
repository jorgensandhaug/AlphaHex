from plants.plant import Plant
import jax
import jax.numpy as jnp

class InvertedPendulum(Plant):
    def __init__(self, length, mass_cart, mass_pendulum, delta_t):
        self.length = length  # Length of the pendulum rod
        self.mass_cart = mass_cart  # Mass of the cart
        self.mass_pendulum = mass_pendulum  # Mass of the pendulum
        self.delta_t = delta_t # for discrete time physics simulation

    def update(self, control_signal, disturbance, plant_state):
        theta = plant_state["theta"]
        theta_dot = plant_state["theta_dot"]

        # previous_state = plant_state.copy()
        # # if theta is very close to pi or -pi, return the same state
        # check = jnp.abs(jnp.abs(theta) - jnp.pi/2) < 1e-6
        

        # Constants
        g = 9.81  # Acceleration due to gravity

        m = self.mass_pendulum
        M = self.mass_cart
        l = self.length
        ct = jnp.cos(theta)
        st = jnp.sin(theta)

        ### OLD, but keeping for reference ###
        # # Equations of motion
        # numerator = control_signal + (m * l * theta_dot**2 * st)
        # denominator = M + m * (1 - ct**2)

        # theta_double_dot = (g * st - ct * numerator / (l * denominator)) - disturbance
        # #theta_double_dot = (g*st - ct*(control_signal + m*l*theta_dot**2*st))/(l*(4/3 - m*ct**2/(M+m)))

        # # find x_double_dot
        # x_double_dot = (control_signal + m*l*theta_double_dot*ct-m*l*theta_dot**2*st)/(M+m)
        ### ------------------------------ ###


        # Calculating theta_double_dot and x_double_dot
        Ft = -control_signal + disturbance
        denominator = l * (-m - M + m * ct**2)

        theta_double_dot = -(Ft * ct + g * m * st + g * M * st - l * m * theta_dot**2 * ct * st) / denominator
        x_double_dot = -(1 / ct) * (Ft * ct - l * m * theta_dot**2 * ct * st + g * m * ct**2 * st) / denominator

        # Update state
        theta_dot += theta_double_dot * self.delta_t
        theta += theta_dot * self.delta_t

        x_dot = plant_state["x_dot"] + x_double_dot * self.delta_t
        x = plant_state["x"] + x_dot * self.delta_t

        # Ensure theta stays within -pi/2 to pi/2
        theta = jnp.clip(theta, -jnp.pi/2, jnp.pi/2)

        plant_state["theta"] = theta
        plant_state["theta_dot"] = theta_dot
        plant_state["x"] = x
        plant_state["x_dot"] = x_dot

        plant_state["output"] = -theta

        return plant_state
        #return jax.lax.cond(check, lambda p_s, s: p_s, lambda p_s, s: s, previous_state, plant_state)
