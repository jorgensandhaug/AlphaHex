import numpy as np
import jax


class CONFIG:
    plantname = "bathtub"  # "cournot" "bath
    use_nn_controller = False

    bathtub_config = {
        "h0": 0.8,
        "A": 1,
        "C": 0.02,
        "epochs": 100,
        "target": 1.0,
        "time_steps": 50,
        "initial_control_signal": 0.0,
        "disturbance_strength": 0.002,
        "pid_lr": 0.05,
        "pid_params": np.array([0.1, 0.1, 0.1]),
        "pid_use_adam_optimizer": False,


        "nn_lr": 0.001,
        "nn_layers": [(3, 20), (20, 20), (20, 20), (20, 1)],
        "nn_activation_functions": [jax.nn.tanh, jax.nn.tanh, jax.nn.tanh, lambda x: x],
        "nn_weight_range": (0.0, 0.01),
        "nn_use_adam_optimizer": True,
        "nn_use_biases": True,
    }

    cournot_config = {
        "q1_0": 0.5,
        "q2_0": 0.5,
        "p_max": 2,
        "c_m": 0.5,
        "epochs": 100,
        "target": 0.2,
        "time_steps": 50,
        "initial_control_signal": 0.0,
        "disturbance_strength": 0.001,
        "pid_lr": 0.3,
        "pid_params": np.array([0.1, 0.1, 0.1]),
        "pid_use_adam_optimizer": False,


        "nn_lr": 0.0003,
        "nn_layers": [(3, 3), (3, 3), (3, 1)],
        "nn_activation_functions": [jax.nn.sigmoid, jax.nn.sigmoid, lambda x: x],
        "nn_weight_range": (0.01, 0.02),
        "nn_use_adam_optimizer": True,
        "nn_use_biases": True,
    }

    pendulum_config = {
        "visual": True,
        "speed_up": 5,

        "theta_0": 0.1,
        "theta_dot_0": 0.0,
        "x_0": 0.0,
        "x_dot_0": 0.0,
        "length": 1.0,
        "mass_cart": 1.0,
        "mass_pendulum": 0.1,
        "delta_t": 0.01,
        "epochs": 100,
        "target": 0.0,
        "time_steps": 100,
        "initial_control_signal": 0.0,
        "disturbance_strength": 0.01,
        "pid_lr": 1,
        # "pid_params": np.array([250.0, 0.0, 2000.0]),
        "pid_params": np.array([10.0, 0.0, 50.0]),
        "pid_use_adam_optimizer": True,


        "nn_lr": 0.2,
        "nn_layers": [(3, 3), (3, 1)],
        "nn_activation_functions": [jax.nn.sigmoid, lambda x: x],
        "nn_weight_range": (0.0, 25.0),
        "nn_use_adam_optimizer": False,
        "nn_use_biases": True,
    }
