from gamesim import run
import random
import os
import numpy as np
from hex import Hex
from termcolor import colored
from players import ACNetPlayer, RandomPlayer, MCTSPlayer




def run_full_tournament(acnet, folder_name, cfg):
    size = cfg["size"]
    use_mcts = cfg["use_mcts_in_topp"]
    num_games = cfg["num_tournament_games"]
    num_simulations_per_move = cfg["num_simulations_per_move"]
    print(f"Loading weights from folder: {folder_name}")

    all_files = os.listdir(folder_name)
    print(all_files)


    # filenames are named from filename = f"{self.save_folder}/anet_weights_{'final' if final else episode}.pth"
    # so we create all players with episode or final as the name
    players = []
    for file in all_files:
        file_type = file.split('.')[1]
        if file_type != 'pth':
            continue
        name = file.split('.')[0].split('weights_')[-1]
        acnet_clone = acnet.copy_and_initialize_weights_from_file(f"{folder_name}/{file}")
        acnet_clone.initialize_net_from_onnx(acnet_clone.compile_model_onnx())
        if use_mcts and num_simulations_per_move is not None:
            player = MCTSPlayer(name, acnet, num_simulations=num_simulations_per_move, policy_epsilon=0.05, sigma=0.0, c_param=1, debug=False, temperature=0)
        else:
            player = ACNetPlayer(acnet_clone, name, use_probs_best_2=cfg["use_probs_best_2_in_topp"])
        players.append(player)

    players.append(RandomPlayer('random'))

    # sort players by sort_order
    players = sorted(players, key=lambda x: x.sort_order())
    print([player.name for player in players])

    if cfg["tournament_last_n_games"]:
        num_games = cfg["tournament_last_n_games"]
        print(f"Only playing last {num_games} games + random player")
        players = [players[0]] + players[-num_games:]

    def play_against(player1, player2, size, num_episodes, starting_player_symbol):
        initial_game_state = Hex(Hex.initialize_game(size), starting_player_symbol)
        policies = {starting_player_symbol: player1, -starting_player_symbol: player2}
        wins = run(initial_game_state, policies, num_episodes, debug=cfg["debug"], use_tqdm=False)
        win_rate = wins/num_episodes
        # win rate percentage rounded to 2 decimals
        return win_rate

    

    win_rates_starting = np.zeros((len(players), len(players)))
    win_rates_second = np.zeros((len(players), len(players)))
    # round robin tournament
    for i in range(len(players)):
        # print a big header with the name of the player that is now going to play against all other players
        print(f"\n\n{colored(f'Player {players[i].name} is now playing against all other players', 'yellow', attrs=['bold'])}\n\n")

        for j in range(i+1, len(players)):
            # Player i starts as 1
            win_rate_i_starting = play_against(players[i], players[j], size, num_games//4, 1)
            win_rate_j_starting = play_against(players[j], players[i], size, num_games//4, 1)

            # Player i starts as -1
            win_rate_i_starting_neg = play_against(players[i], players[j], size, num_games//4, -1)
            win_rate_j_starting_neg = play_against(players[j], players[i], size, num_games//4, -1)

            avg_win_rate_i_starting = (win_rate_i_starting + win_rate_i_starting_neg) / 2
            avg_win_rate_j_starting = (win_rate_j_starting + win_rate_j_starting_neg) / 2

            wrp_i = f"{round(avg_win_rate_i_starting*100, 2)}%"
            wrp_j = f"{round(avg_win_rate_j_starting*100, 2)}%"
            print(f"Win rate of ({colored(wrp_i, 'blue')}) starting player ({colored(players[i].name, 'green')}) vs second player ({colored(players[j].name, 'red')})\t\tWin rate of ({colored(wrp_j, 'blue')}) starting player ({colored(players[j].name, 'green')}) vs second player ({colored(players[i].name, 'red')})")


            win_rates_starting[i, j] = (win_rate_i_starting + win_rate_i_starting_neg) / 2
            win_rates_starting[j, i] = (win_rate_j_starting + win_rate_j_starting_neg) / 2

            win_rates_second[i, j] = 1 - (win_rate_j_starting + win_rate_j_starting_neg) / 2
            win_rates_second[j, i] = 1 - (win_rate_i_starting + win_rate_i_starting_neg) / 2


    # remove diagonal from all win_rates_starting[i] and win_rates_second[i]
    win_rates_starting = np.array([win_rates_starting[i][np.arange(len(win_rates_starting[i])) != i] for i in range(len(win_rates_starting))])
    win_rates_second = np.array([win_rates_second[i][np.arange(len(win_rates_second[i])) != i] for i in range(len(win_rates_second))])

    # get means
    win_rates_starting = np.mean(win_rates_starting, axis=1)
    win_rates_second = np.mean(win_rates_second, axis=1)
    

    # sort players by win rate
    best_win_rates_starting = np.argsort(win_rates_starting)[::-1]
    best_win_rates_second = np.argsort(win_rates_second)[::-1]

    for i in range(len(players)):
        print(f"Player {players[i].name} mean win rate when starting: {win_rates_starting[i]}")
        print(f"Player {players[i].name} mean win rate when second: {win_rates_second[i]}")

    print(f"Best players when starting: {', '.join([players[i].name for i in best_win_rates_starting])}")
    print(f"Best players when second: {', '.join([players[i].name for i in best_win_rates_second])}")

    # save this information to json file. Both best win rates when starting and when second as well as all win rates when starting and when second
    import json
    data = {
        'best_win_rates_starting': [players[i].name for i in best_win_rates_starting],
        'best_win_rates_second': [players[i].name for i in best_win_rates_second],
        'win_rates_starting': win_rates_starting.tolist(),
        'win_rates_second': win_rates_second.tolist()
    }
    with open(f"{folder_name}/win_rates.json", 'w') as f:
        json.dump(data, f)


            