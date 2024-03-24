import numpy as np
from utils import timing_decorator
import random
import torch

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def add_child(self, child_state):
        child_node = Node(child_state, self)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.value += result
        # if result == 1:
        #     if self.state.player_turn == 1:
        #         self.value += 1

        # else:
        #     if self.state.player_turn == -1:
        #         self.value += 1


    def is_fully_expanded(self): # TODO: make this efficient
        legal_actions = self.state.get_legal_actions()
        return len(self.children) == len(legal_actions) and len(self.children) > 0

    def state_value_to_self(self):
        val = (self.value / self.visits + 1) / 2 # between 0 and 1# if 0, that means its good for min player, if 1, that means its good for max player
        if self.state.is_max_player_turn():
            return val
        else:
            return 1 - val

    def child_choices(self, c_param=1):
        is_max_player = self.state.is_max_player_turn()
        log_self_visits = np.log(self.visits)

        q_values = np.array([1-child.state_value_to_self() for child in self.children])

        child_visits = np.array([child.visits for child in self.children])
        exploring_bonuses = c_param * np.sqrt(log_self_visits / (1+child_visits))
        # print(f"Exploring bonuses: {exploring_bonuses}")
        # print(q_values)
        # if is_max_player:
        #     choices_weights = q_values + exploring_bonuses
        # else:
        #     choices_weights = q_values - exploring_bonuses
        #     choices_weights = -choices_weights # flip the sign to make it a maximization problem

        if not np.all(q_values >= 0):
            print(f"Child values: {q_values}")
            print(f"Exploring bonuses: {exploring_bonuses}")
            print(f"Self value: {self.value}")
            print(f"Self visits: {self.visits}")
            print(f"Child visits: {child_visits}")
            print(f"Log self visits: {log_self_visits}")
            print(f"Child values + exploring bonuses: {q_values + exploring_bonuses}")
            raise Exception("Child values are not all positive")
        choices_weights = q_values + exploring_bonuses
            
        return choices_weights

    
    def best_child(self, c_param=1, epsilon=0):
        # if random.random() < epsilon:
        #     return random.choice(self.children)
        choices_weights = self.child_choices(c_param=c_param)
        return self.children[np.argmax(choices_weights)]


    # def action_probs_one_hot_choices(self):
    #     choices_weights = self.child_choices(c_param=c_param)
    #     # print(f"Choices weights: {choices_weights}")
    #     size = self.state.action_state_space_size()
    #     action_probs = np.zeros(size)

    #     best_action = self.children[np.argmax(choices_weights)].state.last_action
    #     action_index = self.state.action_to_action_index(best_action)

    #     action_probs[action_index] = 1
    #     return action_probs

    # def action_probs_one_hot_visits(self):
    #     size = self.state.action_state_space_size()
    #     action_probs = np.zeros(size)

    #     best_action = self.children[np.argmax([child.visits for child in self.children])].state.last_action
    #     action_index = self.state.action_to_action_index(best_action)

    #     action_probs[action_index] = 1
    #     return action_probs



    # def action_probs_distribution_choices(self, c_param=0):
    #     choices_weights = self.child_choices(c_param=c_param)
    #     size = self.state.action_state_space_size()
    #     action_probs = np.zeros(size)


    #     # get the last_action of each child
    #     for i, child in enumerate(self.children):
    #         action = child.state.last_action
    #         action_index = child.state.action_to_action_index(action)
    #         # print(f"Action {i}: {action}, index: {action_index}")
    #         action_probs[action_index] = choices_weights[i]

    #     # # use softmax to normalize the probabilities
    #     # action_probs = ANET.softmax(action_probs) # TODO: figure out whether this is best


    #     action_probs = action_probs / np.sum(action_probs)
    #     return action_probs

    def action_probs_distribution_visits(self, temperature=1):
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
        return self.action_probs_distribution_visits(temperature=temperature)

    def is_leaf(self):
        return len(self.children) == 0

    def choose_move_stochastic(self, temperature=1):
        # choose move based on the number of visits to each child
        children_visits = np.array([child.visits for child in self.children])
        # print(children_visits)
        # normalize the visits first
        children_visits = children_visits / np.sum(children_visits)
        children_visits = children_visits ** (1/temperature)
        visits_distribution = children_visits / np.sum(children_visits)
        child = np.random.choice(self.children, p=visits_distribution)
        # if self.state.is_starting_position() and (self.state.action_to_action_index(child.state.last_action) == 0 or self.state.action_to_action_index(child.state.last_action) == 3):
        #     # print in big red text ERROR
        #     # use \033[91m for red text
        #     print(f"\033[91m WARNING: The root node is choosing a move that is not the most visited one. This should not happen. \033[0m")
        #     print(self.state)
        #     print(f"ACTION CHOSEN: {child.state.last_action}")

        return child

    def choose_move_random(self):
        return random.choice(self.children)

    def choose_move_deterministic(self):
        # choose move based on the number of visits to each child
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
    def __init__(self, state, policy, policy_epsilon, sigma=0.0, c_param=1, debug=False, temperature=1):
        self.root = Node(state)
        self.policy = policy
        self.policy_epsilon = policy_epsilon
        self.debug = debug
        self.sigma = sigma
        self.c_param = c_param
        self.temperature = temperature

    def print_debug(self, message, color):
        if self.debug:
            print(f"{color}{message}{Colors.ENDC}")

    # @timing_decorator("MCTS SELECTION", print_interval=100000)
    def selection(self):
        current_node = self.root
        while current_node.is_fully_expanded():
            #TODO: consider adding random prob for not fully expanded nodes
            current_node = current_node.best_child(c_param=self.c_param, epsilon=self.policy_epsilon)
            # self.print_debug(f"Selecting node with state: {current_node.state}", Colors.CYAN)
        return current_node

    def expansion(self, node):
        actions = node.state.get_legal_actions()

        # Convert the children's last actions into a set for efficient lookup
        existing_actions = set(child.state.last_action for child in node.children)

        # Shuffle the actions to make it more random
        random.shuffle(actions)

        for action in actions:
            if action not in existing_actions:
                new_state = node.state.perform_action(action)
                return node.add_child(new_state)

        return node  # if all possible expansions are already children

    # @timing_decorator("MCTS ROLLOUT", print_interval=100000)
    def rollout(self, node):
        current_state = node.state.clone() # TODO:careful with this if later change to saving rollout states
        # self.print_debug(f"Performing rollout", Colors.YELLOW)
        while not current_state.has_winning_path(-current_state.player_turn):
            # print(f"State before: {current_state}")
            action = self.policy.get_action(current_state, epsilon=self.policy_epsilon)
            # print(f"Action: {action}")
            current_state = current_state.perform_action(action)
            # print(f"State after: {current_state}")
        result = current_state.get_result()
        # self.print_debug(f"Rollout result: {result}", Colors.YELLOW)
        return result

    def backpropagation(self, node, result):
        # self.print_debug(f"Backpropagating result: {result}", Colors.BLUE)
        while node is not None:
            node.update(result)
            node = node.parent

    # @timing_decorator("MCTS BEST_ACTION", print_interval=100)
    def best_action(self, num_simulations, return_actions_probs=False, return_state_action_value=False, action_prob_specific_temperature=None):
        for _ in range(num_simulations):
            # self.print_debug(f"Simulation {_+1}", Colors.HEADER)
            v = self.selection()
            if not v.state.is_terminal():
                v = self.expansion(v)

                if self.sigma > 0 and random.random() < self.sigma:
                    # then get value from anet
                    simulation_result = self.policy.get_value(v.state)
                else:
                    simulation_result = self.rollout(v)

            else:
                simulation_result = v.state.get_result()

            self.backpropagation(v, simulation_result)

        # Choose the best action
        best_child = self.root.choose_move_stochastic(temperature=self.temperature)
        if random.random() < self.policy_epsilon:
            best_child = self.root.choose_move_random()

        best_action = best_child.state.last_action
        
        # self.print_debug(f"Best action determined by mcts: {best_action}", Colors.RED)

        return_value = (best_action, best_child)

        if return_actions_probs:
            return_value = return_value + (self.root.action_probs(temperature=self.temperature if action_prob_specific_temperature is None else action_prob_specific_temperature),)

        if return_state_action_value:
            return_value = return_value + (self.root.state_value_to_self(),) # TODO: might use best_child.state_value_to_self() instead

        return return_value



