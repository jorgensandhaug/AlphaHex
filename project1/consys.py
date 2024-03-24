import numpy as np
import jax
import optax
import jax.numpy as jnp

import matplotlib.pyplot as plt


import pygame
import math
import time

# from jax.config import config
# config.update("jax_debug_nans", True) 


# CONSYS class that includes both the controller and the plant
class CONSYS:
    def __init__(self, controller, plant, optimizer):
        self.controller = controller
        self.plant = plant
        self.optimizer = optimizer

    def loss_func(self, params, config, initial_state, plant, controller):
        state = self._forward(config, initial_state, params, plant, controller)

        mse = jnp.sum(jnp.square(state["error_history"]))
        return mse

    def _forward(self, config, initial_state, params, plant, controller, extra_state_history_names=[]):
        control_signal = config['initial_control_signal']

        time_steps = config['time_steps']

        disturbance_vector = initial_state["disturbance_vector"]

        s = initial_state.copy()


        s["control_signal"] = control_signal
        s["error_history"] = jnp.zeros(time_steps)
        s["output_history"] =  jnp.zeros(time_steps)
        s["control_signal_history"] = jnp.zeros(time_steps)
        
        for name in extra_state_history_names:
            s[f"{name}_history"] = jnp.zeros(time_steps)


        def body_func(i, s):
            s["plant_state"] = plant.update(s["control_signal"], disturbance_vector[i], s['plant_state'])
            output = s["plant_state"]["output"]
            
            for name in extra_state_history_names:
                s[f"{name}_history"] = s[f"{name}_history"].at[i].set(s["plant_state"][name])

            error = config['target'] - output
            s["error_history"] = s["error_history"].at[i].set(error)
            s["output_history"] = s["output_history"].at[i].set(output)

            # get U value from controller
            s["control_signal"] = controller.forward(params, s["error_history"], i)

            s["control_signal_history"] = s["control_signal_history"].at[i].set(s["control_signal"])
            return s

        s = jax.lax.fori_loop(0, time_steps, body_func, s)

        return s
    def update_parameters(self, loss_gradient, config, initial_state, params, opt_state):
        mse, grad = loss_gradient(params, config, initial_state, self.plant, self.controller)
         
        if config["use_adam_optimizer"]:
            updates, opt_state = self.optimizer.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
        else:
            params = self.controller.update_parameters(params, grad, config['learning_rate'])
            opt_state = None

        return params, opt_state
            

    def test_system(self, config, initial_state, params, extra_state_history_names=[]):
        state = self._forward(config, initial_state, params, self.plant, self.controller, extra_state_history_names)
        mse = jnp.sum(jnp.square(state["error_history"]))
        state["mse"] = mse
        return state
    
    def run(self, config, initial_state, params, opt_state, callbacks=[], extra_state_history_names=[]):
        loss_gradient = jax.value_and_grad(self.loss_func, argnums=0)
        key = initial_state['key']
        for i in range(config['epochs']):
            # make disturbance vector
            key, subkey = jax.random.split(key, 2)
            if "disturbance_range" in config.keys():
                initial_state["disturbance_vector"] = jax.random.uniform(subkey, (config["time_steps"],), minval=config["disturbance_range"][0], maxval=config["disturbance_range"][1])
            elif "disturbance_strength" in config.keys():
                initial_state["disturbance_vector"] = jax.random.uniform(subkey, (config["time_steps"],)) * config["disturbance_strength"]
            else:
                raise Exception("No disturbance range or strength specified")

            # one run to get the current state and mse
            s = self.test_system(config, initial_state, params, extra_state_history_names)

            # call all the callbacks with the current state and parameters
            for callback in callbacks:
                callback(s, params, False)

            print(f"Epoch {i}")
            params, opt_state = self.update_parameters(loss_gradient, config, initial_state, params, opt_state)
            
            print(f"\033[92mOpt state: {opt_state}\033[0m")
            

        s = self.test_system(config, initial_state, params, extra_state_history_names)
        for callback in callbacks:
            callback(s, params, True)