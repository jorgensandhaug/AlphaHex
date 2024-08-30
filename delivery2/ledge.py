from mcts import MonteCarloTreeSearch
import random
from tqdm import tqdm

class Ledge:
    def __init__(self, board, player_turn):
        self.board = board
        self.player_turn = player_turn
        self.last_action = None  # Action format: (coin_type, from_index, to_index)

    def is_max_player_turn(self):
        return self.player_turn == 1

    def get_legal_actions(self):
        legal_actions = []
        # Add action to remove the coin at the ledge (if any)
        if self.board[0] != 0:
            legal_actions.append((self.board[0], 0, -1))

        # Add actions to move coins to the left
        for i in range(1, len(self.board)):
            if self.board[i] != 0:
                for j in range(i - 1, -1, -1):
                    if self.board[j] == 0:
                        legal_actions.append((self.board[i], i, j))
                    else:
                        break  # Can't jump over coins

        return legal_actions

    def perform_action(self, action):
        coin_type, from_index, to_index = action
        new_state = self.clone()
        new_state.player_turn *= -1
        new_state.last_action = action

        new_state.board[from_index] = 0  # Remove coin from the original position

        if to_index != -1:
            new_state.board[to_index] = coin_type  # Place coin at the new position

        return new_state

    def clone(self):
        return Ledge(self.board[:], self.player_turn)

    def is_terminal(self):
        return self.board[0] == 2  # Game ends if the gold coin is at the ledge

    def get_result(self):
        # Winner is the one to move
        return 1 if self.player_turn == 1 else -1

    def clone(self):
        return Ledge(self.board[:], self.player_turn)

    def __str__(self):
        return f"Board: {self.board}, Player Turn: {self.player_turn}, Last Action: {self.last_action}"


def random_policy(state):
    actions = state.get_legal_actions()
    return random.choice(actions)

if __name__ == "__main__":
    # Define initial board configuration
    initial_board = [1, 0, 0, 2, 0, 1]
    starting_player = 1
    player_turn = starting_player

    num_simulations = 60
    num_episodes = 1000
    debug = False

    wins = 0

    for episode in tqdm(range(num_episodes)):
        ledge_game = Ledge(initial_board, player_turn)

        mcts = MonteCarloTreeSearch(ledge_game, random_policy, debug=debug)
        while not ledge_game.is_terminal():
            if debug:
                print(f"\nCurrent state: {ledge_game}")
                print(f"Player {ledge_game.player_turn}'s turn.")

            best_action, best_child = mcts.best_action(num_simulations)

            ledge_game = ledge_game.perform_action(best_action)
            if debug:
                print(f"Player {ledge_game.player_turn * -1} performs action {best_action}, resulting in {ledge_game}")

            if ledge_game.is_terminal():
                if debug:
                    print(f"Player {ledge_game.player_turn * -1} wins the game!")
                if ledge_game.player_turn * -1 == starting_player:
                    wins += 1
                break

            mcts.root = best_child
            mcts.root.parent = None

    print(f"Player 1 won {wins} out of {num_episodes} games. Win rate: {wins/num_episodes}")
