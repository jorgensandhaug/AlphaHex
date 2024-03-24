import jax
import jax.numpy as jnp
import optax
from consys import CONSYS
import numpy as np

import matplotlib.pyplot as plt

from plants.pendulum import InvertedPendulum
from plants.bathtub import Bathtub
from plants.cournot import CournotSystem
from controllers.pid import PIDController
from controllers.nn import NNController

import pygame
import math
import time

from config import CONFIG

# from jax.config import config
# config.update("jax_debug_nans", True) 

if __name__ == "__main__":
    extra_state_history_names = []
    mse_history = []
    def mse_plot_callback(state, params, done):
        mse_history.append(state["mse"])

        print(f"\033[91mMSE: {state['mse']} \033[0m")
        # pretty print big yellow text
        print(f"\033[93m")
        print(f"Params: {params}")
        print(f"State: {state['plant_state']}")
        print(f"Done: {done}")
        formatted_error_history = np.array([f"{error:.6f}" for error in state['error_history']])
        print(f"Error history: {formatted_error_history}")
        print(f"\033[0m")

        if done:
            plt.plot(mse_history)
            plt.xlabel("Epoch")
            plt.ylabel("MSE")
            plt.show()
        return
    
    pid_history = []
    def pid_plot_callback(state, params, done):
        pid_history.append(params)
        if done:
            plt.plot(pid_history)
            plt.xlabel("Epoch")
            plt.ylabel("PID Parameters")
            plt.legend(["Kp", "Ki", "Kd"])
            plt.show()
        return

    

    callbacks = [mse_plot_callback]
    if not CONFIG.use_nn_controller:
        callbacks.append(pid_plot_callback)



    # 5.1 Bathub
    if CONFIG.plantname == "bathtub":
        config = CONFIG.bathtub_config

        state = {'plant_state': {"output": config['h0']}, 'key': jax.random.PRNGKey(0)}
        plant = Bathtub(config['A'], config['C'])

    # 5.2 Cournot
    elif CONFIG.plantname == "cournot":
        config = CONFIG.cournot_config
        state = {'plant_state': {"q1": config['q1_0'], "q2": config['q2_0'], 'output': 0.0}, 'key': jax.random.PRNGKey(0)}
        

        plant = CournotSystem(config['p_max'], config['c_m'])
        # print(f"Max possible profit, (best_q, profit): {plant.max_possible_profit(config['q2_0'])}, Target: {config['target']}")


    # 5.3 Pendulum
    elif CONFIG.plantname == "pendulum":
        config = CONFIG.pendulum_config
        
        state = {'plant_state': {"theta": config['theta_0'], "theta_dot": config['theta_dot_0'], 'output': config["theta_0"], "x": config['x_0'], "x_dot": config['x_dot_0']}, 'key': jax.random.PRNGKey(0)}

        extra_state_history_names = ["x", "x_dot"]
        
        plant = InvertedPendulum(config['length'], config['mass_cart'], config['mass_pendulum'], config['delta_t'])


        if config["visual"]:
            # Initialize pygame
            pygame.init()

            # Set the dimensions of the screen
            screen_width, screen_height = 800, 600
            screen = pygame.display.set_mode((screen_width, screen_height))

            # Set the pendulum parameters
            pendulum_length = 100
            pendulum_radius = 10

            def draw(output, state, i, done):
                # Clear the screen
                screen.fill((255, 255, 255))

                # Calculate the pendulum's position
                x = screen_width / 2 + pendulum_length * math.cos(-output + math.pi / 2)
                y = screen_height / 2 - pendulum_length * math.sin(-output + math.pi / 2)

                # Draw the pendulum
                pygame.draw.circle(screen, (0, 0, 0), (int(x), int(y)), pendulum_radius)

                # Draw the pivot point
                pygame.draw.circle(screen, (0, 0, 0), (screen_width // 2, screen_height // 2), pendulum_radius//2)

                # draw the line
                pygame.draw.line(screen, (0, 0, 0), (screen_width // 2, screen_height // 2), (int(x), int(y)))

                
                def display_text(text, y):
                    font = pygame.font.Font('freesansbold.ttf', 32)
                    text = font.render(text, True, (0, 0, 0))
                    textRect = text.get_rect()
                    textRect.center = (screen_width // 2, y)
                    screen.blit(text, textRect)

                # display mse on screen
                if "mse" in state:
                    display_text(f"MSE: {state['mse']}", screen_height // 4)

                # display force on screen
                force = state['control_signal_history'][i]
                display_text(f"Force: {force}", screen_height // 8)

                # display x on screen
                x = state['x_history'][i]
                display_text(f"x: {x}", screen_height // 4 * 3)

                # display x_dot on screen
                x_dot = state['x_dot_history'][i]
                display_text(f"x_dot: {x_dot}", screen_height // 8 * 5)

                # display max force on screen
                display_text(f"Max force: {jnp.max(state['control_signal_history'])}", screen_height // 8 * 3)


                # Update the screen
                pygame.display.flip()

                # Add a delay to control the speed of the animation
                pygame.time.wait(int(1000/config["speed_up"]*config['delta_t']))

            def callback(state, params, done):
                # print(f"Error history: {state['error_history'].astype(float)}")
                #print(f"Output history: {state['output_history'].astype(float)}")
                # print(f"Control signal history: {state['control_signal_history'].astype(float)}")

                
                # Iterate through the output history
                for i, output in enumerate(state['output_history']):
                    # if output is very close to pi divided by 2, break
                    if np.abs(np.abs(output) - np.pi/2) < 1e-6:
                        break

                    force = state['control_signal_history'][i]
                    
                    x = state['x_history'][i]


                    draw(output, state, i, done)

                time.sleep(0.1)
                return



            def plot_error_history_callback(state, params, done):
                plt.plot(state['error_history'])
                plt.xlabel("Time step")
                plt.ylabel("Error")
                plt.show()
                return

            callbacks.append(callback)
            # callbacks.append(plot_error_history_callback)


    # controller initialization 
    if not CONFIG.use_nn_controller:
        controller = PIDController()
        config["learning_rate"] = config["pid_lr"]
        config["use_adam_optimizer"] = config["pid_use_adam_optimizer"]
        params = config["pid_params"]

    else:
        controller = NNController(config["nn_activation_functions"])

        params = controller.init_params(jax.random.PRNGKey(1), config["nn_layers"], nn_weight_range=config["nn_weight_range"], use_biases=config["nn_use_biases"])
        config["learning_rate"] = config["nn_lr"]
        config["use_adam_optimizer"] = config["nn_use_adam_optimizer"]

    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    consys = CONSYS(controller, plant, optimizer)
    consys.run(config, state, params, opt_state, callbacks, extra_state_history_names)



