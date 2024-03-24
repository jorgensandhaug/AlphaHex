from players import ACNetPlayer, HumanPlayer, MCTSPlayer
from hex import Hex
from gamesim import run

# take in filename of weights as argument from command line
def human_against_acnet(acnet, cfg):
    acnet_onnx = acnet.copy_and_initialize_weights_from_file(cfg['filename'])
    acnet_onnx.initialize_net_from_onnx(acnet_onnx.compile_model_onnx())
    # acnet_onnx.initialize_net_from_jit(acnet_onnx.compile_model_jit())
    
    robo = MCTSPlayer("mcts_acnet", acnet_onnx, cfg.get('num_simulations_human_play', 500)) if cfg['use_mcts_in_human_play'] else ACNetPlayer(acnet_onnx, 'acnet', use_probs_best_2=False)
    human = HumanPlayer('human')
    players = []
    if cfg['human_starting']:
        players.append(human)
        players.append(robo)
    else:
        players.append(robo)
        players.append(human)

    initial_game_state = Hex(Hex.initialize_game(cfg['size']), cfg['starting_player_symbol'])

    policies = {cfg['starting_player_symbol']: players[0], -cfg['starting_player_symbol']: players[1]}
    starting_player_wins = run(initial_game_state, policies, 1, debug=False, use_tqdm=False, pass_previous_actions_to_human_player=True)
    if starting_player_wins:
        print(f"Player {cfg['starting_player_symbol']} wins! ({'human' if cfg['human_starting'] else 'acnet'})")
        
    else:
        print(f"Player {-cfg['starting_player_symbol']} wins! ({'acnet' if cfg['human_starting'] else 'human'})")
