from players import HumanPlayer
from tqdm import tqdm

# game sim to play a game. The game state, policy, number of simulations, number of episodes, and debug are all given in the run function
def run(initial_game_state, policies, num_games, debug, use_tqdm=True, pass_previous_actions_to_human_player=False):
    wins = 0
    starting_player = initial_game_state.player_turn
    # Run the mcts for a certain number of episodes
    iterable = tqdm(range(num_games)) if use_tqdm else range(num_games)
    for _ in iterable:
        # Initialize the mcts class
        game = initial_game_state.clone()
        first_action = None

        previous_action = None
        previous_action_probs = None

        while not game.is_terminal():
            if debug==2:
                print(f"\nCurrent state: {game}")
                print(f"Player {game.player_turn}'s turn.")

            player = game.player_turn


            if pass_previous_actions_to_human_player:
                if isinstance(policies[player], HumanPlayer):
                    best_action = policies[player].get_action(game, previous_action=previous_action, previous_action_probs=previous_action_probs)
                
                else:
                    best_action, previous_action_probs = policies[player].get_action_and_probs(game)
                    previous_action = best_action

            else:
                best_action = policies[player](game, epsilon=0)

            game = game.perform_action(best_action)
            
            if first_action == None:
                first_action = best_action

            if debug==2:
                print(f"Player {player} takes action {best_action} piece(s)")


        if debug:
            print(f"Player {player} wins the game!")
            print(f"Last state: {game}")
            print(f"First action: {first_action}")

        if player == starting_player:
            wins += 1

        first_action = None
    return wins