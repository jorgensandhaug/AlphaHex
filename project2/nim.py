from mcts import MonteCarloTreeSearch
import random
from tqdm import tqdm

class NIM:
    def __init__(self, initial_pieces, max_remove, player_turn):
        self.initial_pieces = initial_pieces
        self.max_remove = max_remove
        self.pieces = initial_pieces
        self.player_turn = player_turn
        self.last_action = None

    def is_max_player_turn(self):
        return self.player_turn == 1 # 1 is max player, -1 is min player

    def get_legal_actions(self):
        # print(self.pieces, self.max_remove)
        return list(range(1, min(self.pieces, self.max_remove) + 1))

    def perform_action(self, action):
        new_state = self.clone()
        new_state.pieces -= action
        new_state.player_turn *= -1
        new_state.last_action = action
        return new_state

    def is_terminal(self):
        return self.pieces == 0

    def get_result(self):
        return 1 if self.player_turn == -1 else -1 # if -1, that means player 1 took the last piece, and therefore won

    def clone(self):
        return NIM(self.pieces, self.max_remove, self.player_turn)

    def __str__(self):
        return f"pieces: {self.pieces}, player_turn: {self.player_turn}, last_action: {self.last_action}"



if __name__ == "__main__":
    # Define a random policy function for the simulations
    def random_policy(state):
        actions = state.get_legal_actions()
        # print(f"Legal actions: {actions}")
        return random.choice(actions)



    # Initialize the NIM game state
    initial_pieces = 10  # You can vary this for different tests
    max_remove = 5  # You can vary this as well
    starting_player = 1  # You can vary this as well
    player_turn = starting_player

    num_simulations = 200

    num_episodes = 400

    debug = False


    wins = 0
    # Run the mcts for a certain number of episodes
    for episode in tqdm(range(num_episodes)):
        nim_game = NIM(initial_pieces, max_remove, player_turn)

        # Initialize the mcts with the NIM game state and the random policy
        mcts = MonteCarloTreeSearch(nim_game, random_policy, debug=debug)
        while not nim_game.is_terminal():
            if debug:
                print(f"\nCurrent state: {nim_game.pieces} pieces remaining.")
                print(f"Player {nim_game.player_turn}'s turn.")

            best_action, best_child = mcts.best_action(num_simulations)

            nim_game = nim_game.perform_action(best_action)
            if debug:
                print(f"Player {nim_game.player_turn * -1} takes {best_action} piece(s), leaving {nim_game.pieces}.")

            if nim_game.is_terminal():
                if debug:
                    print(f"Player {nim_game.player_turn * -1} wins the game!")
                if nim_game.player_turn * -1 == starting_player:
                    wins += 1
                break

            # Set the best child as the new root for the mcts
            mcts.root = best_child
            mcts.root.parent = None  # Detach the new root from its parent
            # mcts.root.state.player_turn = nim_game.player_turn  # Update the player turn
        
    print(f"Player 1 won {wins} out of {num_episodes} games. Win rate: {wins/num_episodes}")

