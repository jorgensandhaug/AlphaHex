import numpy as np
from utils import timing_decorator
import random
import torch

class Node:
    def __init__(self, state, parent=None):
        # Core attributes to manage the state and tree structure.
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def add_child(self, child_state):
        # Create a new node and append it to the children list.
        child_node = Node(child_state, self)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        # Update the value and visit count of this node based on the result of a simulation.
        self.visits += 1
        self.value += result


    def is_fully_expanded(self):
        # Check if all possible actions from the current state have been expanded.
        legal_actions = self.state.get_legal_actions()
        return len(self.children) == len(legal_actions) and len(self.children) > 0

    def state_value_to_self(self):
        # This is the value of the state to the player who is making the move
        val = (self.value / self.visits + 1) / 2 # between 0 and 1# if 0, that means its good for min player, if 1, that means its good for max player
        if self.state.is_max_player_turn():
            return val
        else:
            return 1 - val

    def child_choices(self, c_param=1):
        # Calculate the UCT value for each child node.
        is_max_player = self.state.is_max_player_turn()
        log_self_visits = np.log(self.visits)

        q_values = np.array([1-child.state_value_to_self() for child in self.children])

        child_visits = np.array([child.visits for child in self.children])
        # Exploring bonuses is based on UCT ish
        exploring_bonuses = c_param * np.sqrt(log_self_visits / (1+child_visits))


        choices_weights = q_values + exploring_bonuses
            
        return choices_weights

    
    def best_child(self, c_param=1):
        # Return the child with the highest UCT value.
        choices_weights = self.child_choices(c_param=c_param)
        return self.children[np.argmax(choices_weights)]


    def action_probs_distribution_visits(self, temperature=1):
        # Calculate the action probabilities based on the number of visits to each child.
        size = self.state.action_state_space_size()
        action_probs = np.zeros(size)

        # get the last_action of each child
        for i, child in enumerate(self.children):
            action = child.state.last_action
            action_index = child.state.action_to_action_index(action)
            action_probs[action_index] = child.visits

        # normalize the visits first
        action_probs = action_probs / np.sum(action_probs)
        action_probs = action_probs ** (1/temperature)
        action_probs = action_probs / np.sum(action_probs)
        
        return action_probs


    def action_probs(self, temperature=1):
        # Calculate the action probabilities based on the number of visits to each child.
        return self.action_probs_distribution_visits(temperature=temperature)

    def is_leaf(self):
        # Check if the current node is a leaf node.
        return len(self.children) == 0

    def choose_move_stochastic(self, temperature=1):
        # choose move stochastically based on the number of visits to each child
        children_visits = np.array([child.visits for child in self.children])
        # normalize the visits first
        children_visits = children_visits / np.sum(children_visits)
        children_visits = children_visits ** (1/temperature)
        visits_distribution = children_visits / np.sum(children_visits)
        child = np.random.choice(self.children, p=visits_distribution)

        return child

    def choose_move_random(self):
        # choose move randomly
        return random.choice(self.children)

    def choose_move_deterministic(self):
        # Choose the move with the highest number of visits.
        children_visits = np.array([child.visits for child in self.children])
        child = self.children[np.argmax(children_visits)]
        return child





# ANSI escape sequences for colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class MonteCarloTreeSearch:
    def __init__(self, state, policy, policy_epsilon, sigma=0.0, c_param=1, debug=False, temperature=1, discount_factor=1):
        self.root = Node(state)
        self.policy = policy
        self.policy_epsilon = policy_epsilon
        self.debug = debug
        self.sigma = sigma
        self.c_param = c_param
        self.temperature = temperature
        self.discount_factor = discount_factor

    def print_debug(self, message, color):
        if self.debug:
            print(f"{color}{message}{Colors.ENDC}")

    # @timing_decorator("MCTS SELECTION", print_interval=100000)
    def selection(self):
        # Traverse the tree using the tree policy until a leaf node is found.
        current_node = self.root
        while current_node.is_fully_expanded():
            # Select the child node with the highest UCT value. This is the tree policy.
            current_node = current_node.best_child(c_param=self.c_param)
            # self.print_debug(f"Selecting node with state: {current_node.state}", Colors.CYAN)
        return current_node

    def expansion(self, node):
        # Expand the leaf node by adding a new child node randomly based on the available actions.
        actions = node.state.get_legal_actions()

        # Convert the children's last actions into a set for efficient lookup
        existing_actions = set(child.state.last_action for child in node.children)

        # Shuffle the actions
        random.shuffle(actions)

        for action in actions:
            if action not in existing_actions:
                new_state = node.state.perform_action(action)
                return node.add_child(new_state)

        return node  # if all possible expansions are already children

    # @timing_decorator("MCTS ROLLOUT", print_interval=100000)
    def rollout(self, node):
        # Perform a rollout from the current state until a terminal state is reached. 
        current_state = node.state.clone() 
        # self.print_debug(f"Performing rollout", Colors.YELLOW)
        while not current_state.has_winning_path(-current_state.player_turn):
            action = self.policy.get_action(current_state, epsilon=self.policy_epsilon)
            current_state = current_state.perform_action(action)
        result = current_state.get_result()
        # self.print_debug(f"Rollout result: {result}", Colors.YELLOW)
        return result

    def backpropagation(self, node, result):
        # Backpropagate the result of the simulation up the tree to update the value and visit count of each node.
        # self.print_debug(f"Backpropagating result: {result}", Colors.BLUE)
        discount = 1
        while node is not None:
            node.update(result * discount)
            discount *= self.discount_factor
            node = node.parent

    # @timing_decorator("MCTS BEST_ACTION", print_interval=100)
    def best_action(self, num_simulations, return_actions_probs=False, return_state_action_value=False, action_prob_specific_temperature=None):
        # Perform a number of MCTS simulations starting from the root node, then return the best action based on the number of visits to each child.
        for _ in range(num_simulations):
            # self.print_debug(f"Simulation {_+1}", Colors.HEADER)
            v = self.selection()
            if not v.state.is_terminal():
                v = self.expansion(v)

                # Get result using either the policy with rollout or potentially a critic to get the value of the state immediately
                if self.sigma > 0 and random.random() < self.sigma:
                    # then get value from anet
                    simulation_result = self.policy.get_value(v.state)
                    # print(f"Value head result: {simulation_result}, player turn: {v.state.player_turn}")
                    # This is a bit janky, but its because of how i originally trained the value head; between 0 and 1, the higher, the better for the current player
                    if v.state.is_max_player_turn():
                        simulation_result = simulation_result * 2 - 1
                    else:
                        simulation_result = (1 - simulation_result) * 2 - 1

                else:
                    simulation_result = self.rollout(v)

            else:
                simulation_result = v.state.get_result()

            self.backpropagation(v, simulation_result)

        # In the self play, we may want to choose a random action to explore more possible states that the agent may get into when playing against other agents
        # if temperature is very small, then its just deterministic
        if self.temperature >= 0.05: 
            best_child = self.root.choose_move_stochastic(temperature=self.temperature)
        else:
            # Choose the best action deterministically, this may be wanted e.g. if using MCTS for inference
            best_child = self.root.choose_move_deterministic()
                
        best_action = best_child.state.last_action
        
        # self.print_debug(f"Best action determined by mcts: {best_action}", Colors.RED)

        # In training, these are just used to guide the self-play games played, but not used to train the neural network directly
        return_value = (best_action, best_child)

        # These are the things that are actually used to train the neural network
        if return_actions_probs:
            return_value = return_value + (self.root.action_probs(temperature=self.temperature if action_prob_specific_temperature is None else action_prob_specific_temperature),)

        if return_state_action_value:
            return_value = return_value + (self.root.state_value_to_self(),)

        return return_value



