import random
from tqdm import tqdm
import numpy as np
import networkx as nx
from mcts import MonteCarloTreeSearch, Node
from unionfind import UnionFind
from utils import timing_decorator
from has_winning_path import has_winning_path_cython

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
class bcolors:
    BLUE = '\033[94m'
    RED = '\033[91m'
    #PURPLE = '\033[35m'
    PURPLE = '\033[38;5;99m'  # Light purple in 256-color mode
    GREEN = '\033[92m'
    ENDC = '\033[0m'


class Hex:
    memoized_paths = {}
    def __init__(self, board, player_turn, legal_actions=None, legal_actions_flat_mask=None):
        self.board = board
        self.board_size = len(self.board)
        self.board_flat = board.flatten()
        self.player_turn = player_turn

        # try to make things more efficient by storing the legal actions instead of calculating them every time
        if legal_actions is None:
            self.legal_actions = list(map(tuple, np.argwhere(self.board == 0)))
        if legal_actions_flat_mask is None:
            self.legal_actions_flat_mask = (self.board.flatten() == 0).astype(np.int32)

        # This was used for memoization of paths, but it was actually slower than recalculating the paths (because the recalculation is very fast in Cython)
        # self.is_memoized = False

    def get_legal_actions(self):
        return self.legal_actions

    def get_legal_actions_flat_mask(self):
        # A flat version of legal actions
        return self.legal_actions_flat_mask

    def state_equals(self, other):
        return np.array_equal(self.board, other.board) and self.player_turn == other.player_turn

    def is_max_player_turn(self):
        return self.player_turn == 1

    def perform_action(self, action):
        # Performs the action from the current state and returns the new state
        i, j = action
        new_board = self.board.copy()
        new_board[i][j] = self.player_turn
        new_state = Hex(new_board, self.other_player(self.player_turn))
        new_state.last_action = action
        return new_state

    def get_result(self):
        # Returns the result of the game. If the game is not over, returns 0. If player 1 wins, returns 1. If player -1 wins, returns -1.

        # Use memoization to avoid recalculating paths
        # if self.is_memoized:
        #     result = Hex.memoized_paths.get(self.board_tuple)
        # else: 
        result = 1 if self.has_winning_path(1) else -1 if self.has_winning_path(-1) else 0
            # Hex.memoized_paths[self.board_tuple] = result
            # self.is_memoized = True
        return result

    def is_terminal(self):
        # Returns True if the game is over, False otherwise
        return self.get_result() != 0

    # @timing_decorator("has_winning_path_cython", print_interval=1000)
    def has_winning_path(self, player):
        # Check if the player has a winning path
        b = self.board
        size = len(self.board)

        # set dtype to np int 32
        b = b.astype(np.int32)
        wp = has_winning_path_cython(b, size, player)
        return wp

    @staticmethod
    def initialize_game(size):
        # use numpy array for faster indexing
        return np.zeros((size, size), dtype=np.dtype("i4"))

    @staticmethod
    def other_player(player):
        # Returns the other player
        return -player

    def clone(self):
        # Returns a copy of the current state
        return Hex(self.board.copy(), self.player_turn)


    def __str__(self):
        return self.tostr()

    def tostr(self, last_action=None, probs_array=None) -> str:
        """Horrible function that creates a colored printable representation of the board

        Returns:
            str: The colored board
        """
        if probs_array is not None:
            # if 1d, reshape to 2d
            if probs_array.shape[0] == self.board_size**2:
                probs_array = probs_array.reshape(self.board_size, self.board_size)
            def rgb_to_ansi(r, g, b):
                """Convert RGB values to the closest ANSI color code."""
                if r == g and g == b:
                    if r < 8:
                        return 16
                    if r > 248:
                        return 231
                    return round(((r - 8) / 247) * 24) + 232
                return 16 + (36 * round(r / 255 * 5)) + (6 * round(g / 255 * 5)) + round(b / 255 * 5)

            values = probs_array
            values = (values - values.min()) / (values.max() - values.min())
            # make small differences more visible
            values = values**10


            # Apply color to each individual element
            colored_values = np.empty(values.shape, dtype=object)
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    r = int((1 - values[i, j]) * 255)  # Red decreases
                    g = int(values[i, j] * 255)  # Green increases
                    b = 0  # Blue is always 0
                    color_code = rgb_to_ansi(r, g, b)
                    colored_values[i, j] = f"\033[38;5;{color_code}m0\033[0m"

            values = colored_values.reshape(self.board_size, self.board_size)


        se_diag_starts = [(0, i) for i in range(
            self.board_size-1, -1, -1)] + [(i, 0) for i in range(1, self.board_size)]
        diag_lengths_top = [i for i in range(1, self.board_size)]
        diag_lengths = diag_lengths_top + \
            [self.board_size] + diag_lengths_top[::-1]
        diags = list(map(lambda t: [(t[0][0] + i, t[0][1] + i)
                     for i in range(t[1])], list(zip(se_diag_starts, diag_lengths))))
        res = " "*(self.board_size) + \
            f' {bcolors.PURPLE}#{bcolors.ENDC}' + '\n'
        for i, diag in enumerate(diags):
            if probs_array is not None:
                diag_string = " ".join(map(lambda t: values[t[0], t[1]], diag))
            else:
                diag_string = " ".join(map(lambda t: f'{bcolors.GREEN}0{bcolors.ENDC}' if (last_action and last_action[0]==t[0] and last_action[1]==t[1]) else f'{bcolors.BLUE}0{bcolors.ENDC}' if self.board[t[0], t[1]]==1 else f'{bcolors.RED}0{bcolors.ENDC}' if self.board[t[0], t[1]]==-1 else "0", diag))
            prefix = f' {bcolors.RED}#{bcolors.ENDC}' if i < self.board_size - \
                1 else f' {bcolors.PURPLE}#{bcolors.ENDC}' if i == self.board_size-1 else f' {bcolors.BLUE}#{bcolors.ENDC}'
            suffix = f'{bcolors.BLUE}#{bcolors.ENDC}' if i < self.board_size - \
                1 else f'{bcolors.PURPLE}#{bcolors.ENDC}' if i == self.board_size-1 else f'{bcolors.RED}#{bcolors.ENDC}'
            res += " "*((self.board_size*2-1-len(diag)*2+1)//2) + \
                prefix + diag_string + suffix + '\n'
        res += (" "*(self.board_size) +
                f' {bcolors.PURPLE}#{bcolors.ENDC}' + '\n')
        #self.plot()
        return "\n"+res


    @staticmethod
    def test_has_winning_path():
        # Testing function used during development
        def print_board(board):
            for row in board:
                print(' '.join(str(cell) for cell in row))
            print()

        # Test cases
        test_cases = [
            ([[0, 0, 0], [0, 1, 0], [-1, 0, 0]], 0),
            ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 0),
            ([[0, 0, 1], [0, 1, 0], [1, 0, 0]], 1),
            ([[-1, -1, -1], [0, 0, 0], [0, 0, 0]], 0),
            ([[1, 1, 1], [0, 0, 0], [0, 0, 0]], 1),
            ([[1, 0, 0], [1, 0, 0], [1, 0, 0]], 0),
            ([[0, 0, -1], [1, -1, 1], [-1, 1, 0]], -1),
            ([[-1, 0, -1], [1, -1, -1], [-1, 1, 1]], -1),
            ([[-1, 0, -1], [0, -1, 1], [0, 1, -1]], 0),
            ([[1, -1, 0], [1, -1, 0], [1, -1, 0]], -1),
            ([[1, -1, 1, -1], [1, -1, 1, -1], [1, -1, 1, 0], [1, 1, 1, -1]], 0),
            ([[1, -1, 1, 1], [1, -1, 1, -1], [0, 1, 0, -1], [0, -1, 1, -1]], 0),
            ([[1, -1, 1, 1], [1, -1, 1, -1], [1, 1, 0, -1], [0, -1, 1, -1]], 1)
        ]

        for board, winner in test_cases:
            game = Hex(board, 1)  # Player turn doesn't matter for testing has_winning_path
            print_board(board)
            if game.has_winning_path(1):
                if winner != 1:
                    raise Exception("Player 1 has a winning path but shouldn't")
                print("Player 1 has a winning path")
                continue
            if game.has_winning_path(-1):
                if winner != -1:
                    raise Exception("Player -1 has a winning path but shouldn't")
                print("Player -1 has a winning path")
                continue

            if winner != 0:
                raise Exception("No player has a winning path but there should be one")

    def nn_representation(self):
        # Simple flat representation of the board with the player turn as the first element (0 or 1) # This may not be optimal, but the focus of this project has mainly been on the convolutional neural network instead

        b = self.board.flatten() 
        p = np.array([(self.player_turn+1)/2])
        return np.concatenate((p, b))

    def conv_representation(self):
        # stack of 3 boards, one for each players pieces, and one all 1s if player is 1, all 0s if player is -1
        enc = np.stack(
            (
            self.board == 1, 
            self.board == -1, 
            #self.board == 0,
            np.ones_like(self.board) if self.player_turn == 1 else np.zeros_like(self.board)
            ),
            axis=0).astype(np.float32) 
        return enc


    def nn_representation_size(self):
        # Add 1 for the player turn
        return len(self.board.flatten()) + 1 

    
    def action_index_to_action(self, action_index):
        # mapping from action index to action tuple
        i = action_index // len(self.board)
        j = action_index % len(self.board)
        return (i, j)

    def action_state_space_size(self):
        return len(self.board) * len(self.board)

    def action_to_action_index(self, action):
        # mapping from action tuple to action index
        # action is a tuple
        i, j = action
        return i * len(self.board) + j

    def is_starting_position(self):
        # Returns True if the board is empty, False otherwise
        return np.all(self.board == 0)

    @staticmethod
    def example_net_input(size):
        # Returns an example input for the neural network
        p1 = Hex(Hex.initialize_game(size), 1)
        # set the board to a random state, each cell is either 0 or 1 or -1
        p1.board = np.random.choice([-1, 0, 1], size=(size, size))
        return p1
        

    
    def get_diags(self) -> list[tuple[int, int]]:
        # Helper function for visualizing the board
        se_diag_starts = [(0, i) for i in range(
            self.board_size-1, -1, -1)] + [(i, 0) for i in range(1, self.board_size)]
        diag_lengths_top = [i for i in range(1, self.board_size)]
        diag_lengths = diag_lengths_top + \
            [self.board_size] + diag_lengths_top[::-1]
        return list(map(lambda t: [(t[0][0] + i, t[0][1] + i)
                     for i in range(t[1])], list(zip(se_diag_starts, diag_lengths))))


    def plot(self) -> None:
        # Ugly function for plotting the board

        plt.figure(2)
        plt.clf()
        ## Set axis limits
        plt.xlim(-1,2*self.board_size)
        plt.ylim(2*self.board_size + math.sqrt(3),-math.sqrt(3))

        G=nx.Graph()
        diags = self.get_diags()
        color_map = []
        size_map = [100]*(self.board_size*self.board_size)
        for i, diag in enumerate(diags):
            for j, pos in enumerate(diag):
                G.add_node(str(pos),pos=(((self.board_size*2-1-len(diag)*2+1)//2)/2+j, i*math.sqrt(3)/2))
                piece = self.board[pos[0], pos[1]]
                color_map.append('red' if piece == 1 else 'blue' if piece==-1 else 'yellow')

        edges = G.edges()
        colors = [G[u][v]['color'] for u,v in edges]
        weights = [G[u][v]['weight'] for u,v in edges]
        pos=nx.get_node_attributes(G,'pos')
        nx.draw(G,pos, node_color = color_map,  edge_color=colors, width=weights, node_size = size_map)

        ## Draw the border lines
        (x1, y1) = (((self.board_size*2-2)//2)/2, -(math.sqrt(3)))
        (x2, y2) = (self.board_size, (self.board_size-1)*math.sqrt(3)/2)
        (x3, y3) = (x1, 2*(self.board_size-1)*math.sqrt(3)/2+(math.sqrt(3)))
        (x4, y4) = (-1, y2)
        plt.plot([x1,x2], [y1,y2], color = 'red', linewidth = 10, alpha=0.5)
        plt.plot([x2,x3], [y2,y3], color = 'blue', linewidth = 10, alpha=0.5)
        plt.plot([x3,x4], [y3,y4], color = 'red', linewidth = 10, alpha=0.5)
        plt.plot([x4,x1], [y4,y1], color = 'blue', linewidth = 10, alpha=0.5)

        # plot names of axis, (a, b, c, d,... are the names of the x-axis), (1, 2, 3, 4,... are the names of the y-axis)
        # Define the corners
        corners = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

        # Define the labels
        x_labels = [chr(97+i) for i in range(self.board_size)]
        y_labels = [str(i+1) for i in range(self.board_size)]

        # Define the offset
        start_offset = 0.7
        end_offset = 0.4

        # Interpolate along the lines to place the x labels
        for i, label in enumerate(x_labels):
            x = np.interp(i, [0, self.board_size-1], [x1+start_offset-0.1, x2-end_offset])
            y = np.interp(i, [0, self.board_size-1], [y1+start_offset+0.6, y2-end_offset])
            plt.text(x, y, label, fontsize=12, ha='right', va='bottom', color='black')

        # Interpolate along the lines to place the y labels
        for i, label in enumerate(y_labels):
            x = np.interp(i, [0, self.board_size-1], [x4+start_offset, x1-end_offset])
            y = np.interp(i, [0, self.board_size-1], [y4-start_offset+0.2, y1+end_offset+1])
            plt.text(x, y, label, fontsize=12, ha='right', va='bottom', color='black')


        plt.savefig('current_game.png')



if __name__ == "__main__":
    # Hex.test_has_winning_path()


    # Define a random policy function for the simulations
    class RandomPolicy():
        def get_action(self, state, epsilon=0):
            actions = state.get_legal_actions()
            return random.choice(actions)

    # Initialize the Hex game state
    size = 4  
    starting_player = 1  
    mcts_players = [1, -1]
    player_turn = starting_player

    num_simulations = 500

    num_episodes = 100

    debug = False


    wins = 0
    # Run the mcts for a certain number of episodes
    for episode in tqdm(range(num_episodes)):
        # Initialize the mcts class
        initial_board = Hex.initialize_game(size)
        hex_game = Hex(initial_board, player_turn)
        mcts = MonteCarloTreeSearch(hex_game, RandomPolicy(), policy_epsilon=0.1, debug=debug)

        while not hex_game.is_terminal():
            if debug:
                print(f"\nCurrent state: {hex_game}")
                print(f"Player {hex_game.player_turn}'s turn.")

            player = hex_game.player_turn

            if player in mcts_players:
                best_action, best_child = mcts.best_action(num_simulations)

            else:
                best_action = random_policy(hex_game)
                # new root
                state_after = hex_game.perform_action(best_action)
                best_child = Node(state_after)

            hex_game = hex_game.perform_action(best_action)

            


            if debug:
                print(f"Player {player} takes action {best_action} piece(s)")

            if hex_game.is_terminal():
                if debug:
                    print(f"Player {player} wins the game!")
                if player == starting_player:
                    wins += 1
                break

            # Set the best child as the new root for the mcts
            mcts.root = best_child
            mcts.root.parent = None  # Detach the new root from its parent
        
    print(f"Player 1 won {wins} out of {num_episodes} games. Win rate: {wins/num_episodes}")



