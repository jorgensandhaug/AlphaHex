import numpy as np
from collections import deque
from mcts import MonteCarloTreeSearch
from gamesim import run
import random
import time
from tqdm import tqdm
from players import MCTSPlayer, ACNetPlayer
from multiprocessing import Pool
import torch
import onnx
import onnxruntime as ort


import threading
import io


def run_self_play_wrapper_onnx(args): #THIS WORKS SOO WELL, 30x speedup over jit
    self, best_model_serialized, best_network, num_simulations_per_move, current_self_play_game, iteration, num_iterations = args
    best_network = best_network.copy()
    best_network.initialize_net_from_onnx(best_model_serialized)
    return self.run_self_play(best_network, num_simulations_per_move, current_self_play_game, iteration, num_iterations)

def run_self_play_wrapper(args):
    self, best_network, num_simulations_per_move, current_self_play_game, iteration, num_iterations = args
    return self.run_self_play(best_network, num_simulations_per_move, current_self_play_game, iteration, num_iterations)

class RL:
    def __init__(self, acnet, initial_game_state, cfg):
        self.acnet = acnet
        self.rbuf = deque(maxlen=cfg["replay_buffer_size"])
        self.initial_game_state = initial_game_state.clone()
        self.use_conv_representation = cfg["type_net"] == "cnn"
        self.cfg = cfg
    
    def clear_replay_buffer(self):
        self.rbuf.clear()
    
    def add_to_replay_buffer(self, case):
        self.rbuf.append(case)
    
    
    def train_acnet(self, batch_size):
        # # print (self.rbuf)
        # for x, y in self.rbuf:
        #     print()
        #     print(f"Input size: {len(x)}")
        #     print(f"Output size: {len(y)}")
        #     print(f"Input to ANET: {x}")
        #     print(f"Output from ANET: {y}")

        # use numpy to get batch_size number of random indices from the replay buffer (or all of them if there are less than batch_size)
        # print(f"Number of cases in replay buffer: {len(self.rbuf)}")
        indices = np.random.choice(len(self.rbuf), min(batch_size, len(self.rbuf)), replace=False)
        minibatch = [self.rbuf[i] for i in indices]
        print(f"Training ANET with minibatch of size {len(minibatch)}")
        # Train the ANET using the selected minibatch.
        policy_loss, value_loss = self.acnet.train(minibatch)
        return policy_loss, value_loss

    def get_epsilon(self, episode, num_episodes):
        # linear annealing of epsilon from initial_epsilon to final_epsilon over the course of num_episodes
        frac = min(episode / num_episodes, 1)
        return self.cfg["initial_epsilon"] - (self.cfg["initial_epsilon"] - self.cfg["final_epsilon"]) * frac

    def get_sigma(self, episode, num_episodes):
        # Linear annealing of sigma from 0 to final_sigma over the course of num_episodes
        frac = min(episode / num_episodes, 1)
        return self.cfg["final_sigma"] * frac

    def run_simulations_and_train_policy(self):
        """
        This function runs a simpler training loop as described in the lecture slides of IT3105 AI Programming.
        """

        # Store the loss history
        loss_history = []

        for episode in range(self.cfg["num_episodes"]):
            # initialize the game state
            game_state = self.initial_game_state.clone()
            if self.cfg["alternate_players_during_training"]:
                game_state.player_turn = 1 if episode % 2 == 0 else -1

            # Get the epsilon and sigma values for the current episode
            epsilon = self.get_epsilon(episode, self.cfg["num_episodes"])
            sigma = self.get_sigma(episode, self.cfg["num_episodes"])

            
            # This is just to optimize the speed of the self-play phase
            if self.cfg["optimize_model_for_inference"] == "onnx":
                model_serialized = self.acnet.compile_model_onnx()
                self.acnet.initialize_net_from_onnx(model_serialized)

            # Initialize the MCTS with the current network
            mcts = MonteCarloTreeSearch(game_state, self.acnet, policy_epsilon=epsilon, sigma=sigma, c_param=self.cfg["c_param"])
            
            
            # Run the self-play phase, 1 actual game
            while not game_state.is_terminal():
                # display the game state
                if self.cfg["debug"]==2:
                    print(f"Current game state: {game_state}")

                # Get the best action, best child, action probabilities and state action value given the current game state
                best_action, best_child, action_probs, state_action_value = mcts.best_action(self.cfg["num_simulations_per_move"], return_actions_probs=True, return_state_action_value=True, action_prob_specific_temperature=self.cfg["action_prob_specific_temperature"])

                # Add the case to the replay buffer
                x = game_state.conv_representation() if self.use_conv_representation else game_state.nn_representation()

                # create the case to add to the replay buffer
                case = (x, action_probs, state_action_value)
                self.add_to_replay_buffer(case)

                # Perform the action to get the next game state in the actual game
                game_state = best_child.state

                # Prune off some of the tree and set the new root to the child of the action taken
                mcts.root = best_child
                mcts.root.parent = None # Detach the new root from its parent

                
            print(f"Game {episode} ended with result: {game_state.get_result()}")

            # Before training, make sure that ANET is not using onnx
            if self.cfg["optimize_model_for_inference"] == "onnx":
                self.acnet.use_uncompiled_network()
            
            # Train the ANET using the replay buffer
            policy_loss, value_loss = self.train_acnet(batch_size=self.cfg["batch_size"])
            loss_history.append((policy_loss, value_loss))

            print(f"\033[93m Policy Loss: {policy_loss}, Value Loss: {value_loss} \033[0m")
            # print current epsilon and sigma values
            print(f"\033[93m Epsilon: {epsilon}, Sigma: {sigma} \033[0m")



            # Save the weights of the network every save_interval episodes so we can use them later in topp or human play
            if self.cfg["save_interval"] and episode % self.cfg["save_interval"] == 0:
                self.acnet.save_weights(episode, final=False)
                print(f"Saved weights at episode {episode}")

        self.acnet.save_weights(episode, final=True)
        print(f"Saved weights at episode {self.cfg['num_episodes']}")
        
        return loss_history

    
    def run_iterations(self):
        """
        This function runs a training loop very similar to the one used in AlphaGo Zero. It is based on the following steps:
        1. Self-Play Phase: The best network plays games against itself using MCTS to generate a lot of training data.
        2. Training Phase: The current network is trained using the training data generated in the self-play phase and remaining in the replay buffer.
        3. Evaluation Phase: The current network plays games against the best network to evaluate its performance. Only if the current network wins more than 55% of the games, it becomes the new best network.
        4. Repeat the process for a number of iterations.

        This is based on https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0
        """
        
        if self.cfg["start_training_from_file"]:
            print(f"Starting training from best_network: file {self.cfg['start_training_from_file']}")
            best_network_filename = self.cfg["start_training_from_file"]
            start_iteration = 1+int(best_network_filename.split("_")[-2])

            # initialize the acnet with the weights from the best network
            self.acnet = self.acnet.copy_and_initialize_weights_from_file(best_network_filename)
            
        else:
            best_network_filename = self.acnet.save_weights("initial", final=False)
            start_iteration = 0

        best_network = self.acnet.copy_and_initialize_weights_from_file(best_network_filename)


        # Keep track of time spent in each iteration
        last_time = time.time()
        for iteration in range(start_iteration, self.cfg["num_iterations"]):
            losses = np.array([])
            start_time = time.time()
            print(f"\n--- Iteration {iteration}/{self.cfg['num_iterations']} ---")
            print(f"Time spent on last iteration: {start_time - last_time} seconds")
            last_time = start_time
            print(f"Current best network: {best_network_filename}")
            

            print(f"Starting self-play with {self.cfg['num_self_play_games']} games.")
            # Self-Play Phase
            # Using onnx for self-play is much faster
            if self.cfg["optimize_model_for_inference"] == "onnx":
                best_model_serialized = best_network.compile_model_onnx()
                # run with cfg["mt_threads"] threads for self-play speed up
                with Pool(processes=self.cfg["mt_threads"]) as p:
                    args = [(self, best_model_serialized, best_network, self.cfg['num_simulations_per_move'], current_self_play_game, iteration, self.cfg["num_iterations"]) for current_self_play_game in range(self.cfg['num_self_play_games'])]
                    local_buffers = p.map(run_self_play_wrapper_onnx, args)
                for local_buffer in local_buffers:
                    self.rbuf.extend(local_buffer)

            else:
                best_network.net.eval() # Set the best network to evaluation mode
                with Pool(processes=self.cfg["mt_threads"]) as p:
                    args = [(self, best_network, self.cfg['num_simulations_per_move'], current_self_play_game, iteration, self.cfg["num_iterations"]) for current_self_play_game in range(self.cfg['num_self_play_games'])]
                    local_buffers = p.map(run_self_play_wrapper, args)
                for local_buffer in local_buffers:
                    self.rbuf.extend(local_buffer)
            
            
            # Initialize best network aswell to speed up evaluation
            best_network.initialize_net_from_onnx(best_network.compile_model_onnx())
            print(f"Time spent on self play: {time.time() - start_time} seconds")
            print(f"Starting training with {len(self.rbuf)} cases in replay buffer.")
            # Training Phase
            for epoch in range(self.cfg['num_training_epochs']):
                policy_loss, value_loss = self.train_acnet(self.cfg["batch_size"])
                losses = np.append(losses, [policy_loss, value_loss])
            
                # Evaluation Phase
                # Here, the best_network is used to evaluate the current network
                if epoch % self.cfg["evaluate_interval"] == 0:
                    # compile the network to onnx for evaluation speed up
                    self.acnet.initialize_net_from_onnx(self.acnet.compile_model_onnx())
                    win_ratio = self.evaluate_network(self.cfg["evaluation_games"], best_network, self.cfg['num_simulations_per_move'])
                    self.acnet.use_uncompiled_network()
                    desc = "(1/4 games as starting as 1, 1/4 games as starting as -1, 1/4 games as second as 1, 1/4 games as second as -1)" if self.cfg["alternate_players_during_evaluation"] else "(half games as starting, half games as second)"
                    epoch_str = f"{epoch}".rjust(len(str(self.cfg['num_training_epochs'])))
                    print(f"Evaluating network after {epoch_str} training epochs. Win ratio: {win_ratio:2f} against current best network. {desc}")
                    if win_ratio > 0.55:  # If the new network wins more than 55% of the evaluation games
                        print(f"New best network found at iteration {iteration}, epoch {epoch}")
                        best_network_filename = self.acnet.save_weights(f"{iteration}_{epoch}", final=False)
                        best_network = self.acnet.copy_and_initialize_weights_from_file(best_network_filename)
                        best_network.initialize_net_from_onnx(best_network.compile_model_onnx())

            best_network.use_uncompiled_network()
                        
            

            print(f"Average policy loss for iteration {iteration + 1}: {np.mean(losses[::2])}")
            print(f"Average value loss for iteration {iteration + 1}: {np.mean(losses[1::2])}")

        # Save the final version of the network
        best_network_filename = self.acnet.save_weights(iteration, final=True)
        print(f"Best network final weights saved to {best_network_filename} after {self.cfg['num_iterations']} iterations.")

        return loss_history

    def run_self_play(self, policy, num_simulations_per_move, current_self_play_game, current_rl_iteration, num_rl_iterations):
        local_buffer = []
        game_state = self.initial_game_state.clone()
        if self.cfg["alternate_players_during_training"]:
            game_state.player_turn = 1 if current_self_play_game % 2 == 0 else -1

        epsilon = self.get_epsilon(current_rl_iteration, num_rl_iterations)
        sigma = self.get_sigma(current_rl_iteration, num_rl_iterations)

        mcts = MonteCarloTreeSearch(game_state, policy, policy_epsilon=epsilon, sigma=sigma, c_param=self.cfg["c_param"], temperature=self.cfg["temperature"], discount_factor=self.cfg["discount_factor"])

        while not game_state.is_terminal():
            best_action, best_child, action_probs, state_action_value = mcts.best_action(self.cfg['num_simulations_per_move'], return_actions_probs=True, return_state_action_value=True, action_prob_specific_temperature=self.cfg["action_prob_specific_temperature"])

            # Add the case to the replay buffer
            x = game_state.conv_representation() if self.use_conv_representation else game_state.nn_representation()
            case = (x, action_probs, state_action_value)

            local_buffer.append(case)
            # Get the next state, but keep that part of the tree
            # Prune off some of the tree and set the new root to the child of the action taken
            game_state = best_child.state
            mcts.root = best_child
            mcts.root.parent = None # Detach the new root from its parent
        
        return local_buffer

    def evaluate_network(self, evaluation_games, best_network, num_simulations_per_move):
        # Code to evaluate the new network against the best network

        win_rate_challenger_starting = self.play_evaluation_game(starting=self.acnet, second=best_network, num_games=evaluation_games//2, num_simulations_per_move=num_simulations_per_move)
        win_rate_best_second = 1 - win_rate_challenger_starting

        win_rate_best_starting = self.play_evaluation_game(starting=best_network, second=self.acnet, num_games=evaluation_games//2, num_simulations_per_move=num_simulations_per_move)
        win_rate_challenger_second = 1 - win_rate_best_starting
            
        return (win_rate_challenger_starting + win_rate_challenger_second)/2

    def play_evaluation_game(self, starting, second, num_games, num_simulations_per_move): 
        game_state = self.initial_game_state.clone()
        if self.cfg["use_mcts_in_evaluation"]:
            starting = MCTSPlayer("starting", starting, self.cfg['num_simulations_per_move'], policy_epsilon=0, sigma=0, c_param=self.cfg["c_param"], debug=False, temperature=0)
            second = MCTSPlayer("second", second, self.cfg['num_simulations_per_move'], policy_epsilon=0, sigma=0, c_param=self.cfg["c_param"], debug=False, temperature=0)
        else:
            starting = ACNetPlayer(starting, "starting", use_probs_best_2=self.cfg["use_probs_best_2_in_evaluation"])
            second = ACNetPlayer(second, "second", use_probs_best_2=self.cfg["use_probs_best_2_in_evaluation"])

        if not self.cfg["alternate_players_during_evaluation"]:
            policies = {1: starting, -1: second}
            wins = run(self.initial_game_state.clone(), policies, num_games, debug=False, use_tqdm=False)
            return wins/num_games
    
        else:
            policies = {1: starting, -1: second}
            game_state.player_turn = 1
            wins_1_starting = run(game_state, policies, num_games//2, debug=False, use_tqdm=False)
            
            policies = {1: second, -1: starting}
            game_state.player_turn = -1
            wins_2_starting = run(game_state, policies, num_games//2, debug=False, use_tqdm=False)

            return (wins_1_starting + wins_2_starting)/num_games

        
