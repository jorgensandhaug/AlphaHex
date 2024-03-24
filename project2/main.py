import json
from rl import RL
from hex import Hex
from actor_critic import ACCNN,ACNET
import torch.optim as optim
import torch.nn as nn
from topp import run_full_tournament
from play_against_acnet import human_against_acnet

def load_config(filename):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config

if __name__ == "__main__":
    config = load_config('configs/config.json')
    rlcfg = config[config['training_type']]

    initial_game_state = Hex(Hex.initialize_game(config['size']), config['starting_player'])


    if config['type_net'] == "cnn":
        acnet = ACCNN(
            size=config['size'], 
            in_channels=config['cnn']['in_channels'], 
            num_residual_blocks=config['cnn']['num_residual_blocks'], 
            num_filters=config['cnn']['num_filters'], 
            policy_output_dim=initial_game_state.action_state_space_size(), 
            optimizer_class=getattr(optim, config['cnn']['optimizer']),
            save_folder=f"weights/cnn/{config['size']}_{config['cnn']['in_channels']}_{config['cnn']['num_residual_blocks']}_{config['cnn']['num_filters']}", 
            optimizer_params=config['cnn']['optimizer_params'],
            kernel_size=config['cnn']['kernel_size'],
            device=config['device']
        )

    elif config['type_net'] == "nn":
        acnet = ACNET(
            initial_game_state.nn_representation_size(), 
            initial_game_state.action_state_space_size(), 
            config['nn']['hidden_layers'], 
            [getattr(nn, func_name) for func_name in config['nn']['activation_functions']], 
            optimizer_class=getattr(optim, config['nn']['optimizer']),
            save_folder=f"weights/nn/{config['size']}_{','.join([str(x) for x in config['nn']['hidden_layers']])}_{','.join([x.__name__ for x in [getattr(nn, func_name) for func_name in config['nn']['activation_functions']]])}", 
            optimizer_params=config['nn']['optimizer_params'],
            device=config['device'],
            size=config['size']
        )

    if config['do_human_play']:
        cfg = config['human_against_acnet']
        cfg = {**config, **cfg}
        human_against_acnet(acnet, cfg)

    elif config['do_training']:
        cfg = {**config, **rlcfg}
        cfg["save_interval"] =rlcfg['num_episodes'] // (rlcfg['m']-1) if "m" in rlcfg else None, 
        # rl = RL(acnet, initial_game_state, save_interval=config['num_episodes'] // (config['m']-1), use_conv_representation=config['type_net'] == "cnn")
        rl = RL(
            acnet, 
            initial_game_state, 
            cfg
        )
        if config['training_type'] == "standard_rl":
            rl.run_simulations_and_train_policy(
                num_episodes=config['standard_rl']['num_episodes'], 
                num_simulations=config['standard_rl']['num_simulations'], 
                batch_size=config['standard_rl']['batch_size'], 
            )

        elif config['training_type'] == "alpha_zero_rl":
            rl.run_iterations()


    else:
        cfg = {**config, **rlcfg}
        run_full_tournament(acnet, acnet.save_folder, cfg)