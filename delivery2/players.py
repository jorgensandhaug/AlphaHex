# This file contains the definition of the Player class and its subclasses. The Player class is an abstract class that defines the interface for all player classes. The subclasses of the Player class implement different strategies for playing the game. The subclasses include RandomPlayer, MCTSPlayer, and HumanPlayer. The RandomPlayer class implements a player that selects actions randomly. The MCTSPlayer class implements a player that uses the Monte Carlo Tree Search algorithm to select actions. The HumanPlayer class implements a player that allows a human player to select actions interactively.

import numpy as np
import random
from mcts import MonteCarloTreeSearch, Node


class Player:
    def __init__(self, name):
        self.name = name

    def get_action(self, state, epsilon=0):
        raise NotImplementedError

    def __call__(self, state, epsilon=0):
        return self.get_action(state, epsilon)

    def sort_order(self):
        # sort order for TOPP
        return 0

# Choosees the action based on the policy network
class ACNetPlayer(Player):
    def __init__(self, acnet, name, use_probs_best_2=True):
        super().__init__(name)
        self.acnet = acnet
        self.use_probs_best_2 = use_probs_best_2

    def get_action(self, state, epsilon=0):
        if self.use_probs_best_2:
            return self.acnet.get_action_from_best_2_based_on_probs(state)
        return self.acnet(state, epsilon)

    def get_action_and_probs(self, state):
        best_action, action_probs = self.acnet.get_action_and_probs(state)
        print(f"APROBS: {action_probs.shape}")
        return self.acnet.get_action_and_probs(state)

    def sort_order(self):
        if self.name == 'final':
            return np.inf
        # is it a number?
        elif self.name.isnumeric():
            return int(self.name)

        # else if has form int_int
        elif '_' in self.name:
            ints = self.name.split('_')
            if len(ints) != 2 and not ints[0].isnumeric() and not ints[1].isnumeric():
                return 0
            i1 = int(ints[0])
            i2 = int(ints[1])
            return i1*1000000 + i2
        return 0

# Chooses the action based on the value network in each child state, then chooses the action with the highest value for the current player
class CriticOnlyPlayer(Player):
    def __init__(self, acnet, name):
        super().__init__(name)
        self.acnet = acnet

    def get_action(self, state, epsilon=0):
        legal_actions = state.get_legal_actions()
        action_values_for_other_player = []
        for action in legal_actions:
            next_state = state.perform_action(action)
            if next_state.has_winning_path(state.player_turn):
                return action

            value_for_other_player = self.acnet.get_value(next_state) 
            action_values_for_other_player.append(value_for_other_player)

        best_action_for_current_player = legal_actions[np.argmin(action_values_for_other_player)]
        return best_action_for_current_player

    def get_action_and_probs(self, state):
        legal_actions = state.get_legal_actions()
        action_values_for_other_player = []
        for action_index in range(state.action_state_space_size()):
            action = state.action_index_to_action(action_index)
            if action not in legal_actions:
                # if the action is not legal, then set the value to 1 so that value for current player is 0
                action_values_for_other_player.append(1)
                continue

            next_state = state.perform_action(action)
            if next_state.has_winning_path(state.player_turn):
                return action, np.zeros(len(legal_actions))

            value_for_other_player = self.acnet.get_value(next_state) 
            action_values_for_other_player.append(value_for_other_player)


        # 1- because we want the action with the highest value for the current player, not its children (other player)
        action_probs = 1-np.array(action_values_for_other_player)
        print(f"APROBS: {action_probs.shape}")
            
        best_action_index = np.argmax(action_probs)
        best_action_for_current_player = state.action_index_to_action(best_action_index)
        return best_action_for_current_player, action_probs

    def sort_order(self):
        if self.name == 'final':
            return np.inf
        # is it a number?
        elif self.name.isnumeric():
            return int(self.name)

        # else if has form int_int
        elif '_' in self.name:
            ints = self.name.split('_')
            if len(ints) != 2 and not ints[0].isnumeric() and not ints[1].isnumeric():
                return 0
            i1 = int(ints[0])
            i2 = int(ints[1])
            return i1*1000000 + i2
        return 0
    

# Random player that chooses actions randomly
class RandomPlayer(Player):
    def __init__(self, name):
        super().__init__(name)

    def get_action(self, state, epsilon=0):
        return random.choice(state.get_legal_actions())
    
    def sort_order(self):
        return -np.inf

class RandomPolicy:
    def get_action(self, state, epsilon=0):
        return random.choice(state.get_legal_actions())

# MCTS player that uses the Monte Carlo Tree Search with an acnet to choose actions
class MCTSPlayer(Player):
    def __init__(self, name, acnet, num_simulations, policy_epsilon=0.0, sigma=0.0, c_param=1.4, debug=False, temperature=0, action_prob_specific_temperature=1):
        super().__init__(name)
        self.mcts = MonteCarloTreeSearch(None, acnet, policy_epsilon, sigma, c_param, debug, temperature=temperature)
        self.num_simulations = num_simulations
        self.previous_best_child = None
        self.temperature = temperature
        self.action_prob_specific_temperature = action_prob_specific_temperature

    def get_action(self, state, epsilon=0):
        self.mcts.root = Node(state)
        if self.previous_best_child:
            # check if the current state matches the state of any of the children of the previous best child
            # if it does, then set the root to that child, else set the root to new node with the current state
            for child in self.previous_best_child.children:
                if child.state.state_equals(state):
                    self.mcts.root = child
                    break


        best_action, best_child = self.mcts.best_action(self.num_simulations)
        self.previous_best_child = best_child
        return best_action

    def get_action_and_probs(self, state):
        self.mcts.root = Node(state)
        if self.previous_best_child:
            # check if the current state matches the state of any of the children of the previous best child
            # if it does, then set the root to that child, else set the root to new node with the current state
            for child in self.previous_best_child.children:
                if child.state.state_equals(state):
                    self.mcts.root = child
                    break

        best_action, best_child, action_probs, state_action_value = self.mcts.best_action(self.num_simulations, return_actions_probs=True, return_state_action_value=True, action_prob_specific_temperature=self.action_prob_specific_temperature)
        self.previous_best_child = best_child
        return best_action, action_probs

    def sort_order(self):
        if self.name == 'final':
            return np.inf
        # is it a number?
        elif self.name.isnumeric():
            return int(self.name)

        # else if has form int_int
        elif '_' in self.name:
            ints = self.name.split('_')
            if len(ints) != 2 and not ints[0].isnumeric() and not ints[1].isnumeric():
                return 0
            i1 = int(ints[0])
            i2 = int(ints[1])
            return i1*1000000 + i2
        return 0


# Human player that allows a human player to choose actions interactively
class HumanPlayer(Player):
    def __init__(self, name):
        super().__init__(name)

    def get_action(self, state, previous_action, previous_action_probs):
        print("-------------------"*4)
        if previous_action:
            print(f"Previous action probs: {state.tostr(probs_array=previous_action_probs)}")
            # Now print the action probabilities in the two-dimensional array. Pretty print and round to 2 decimal places.
            print(f"Action probs in 2D array:\n{np.round(previous_action_probs.reshape(state.board_size, state.board_size), 3)}")   
        
            print(f"Current state: {state.tostr(last_action=previous_action)}")
        print(f"Player {state.player_turn}'s turn.")

        state.plot()
        while True:
            action_input = input("Enter action (e.g., a1): ")
            if len(action_input) >= 2 and action_input[0].isalpha() and action_input[1:].isdigit():
                action = self.to_action(state, action_input)
                if action:
                    return action
                else:
                    print("Invalid action. Please try again.")
            else:
                print("Invalid input. Please enter a letter and a number.")


    def sort_order(self):
        return -np.inf
    
    def to_action(self, state, action_input):
        # Convert the letter to a row index
        row = ord(action_input[0].lower()) - ord('a')
        # Convert the number to a column index
        column = int(action_input[1:]) - 1

        if row < 0 or row >= state.action_state_space_size() or column < 0 or column >= state.action_state_space_size():
            return False

        action_index = row * state.board_size + column
        action = state.action_index_to_action(action_index)

        if action not in state.get_legal_actions():
            return False

        return action