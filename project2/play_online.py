# Import and initialize your own actor
import numpy as np
import time
from actor_critic import ACCNN
from players import ACNetPlayer, MCTSPlayer, CriticOnlyPlayer
from hex import Hex
filename = "weights/cnn/7_3_8_32/anet_weights_446_0.pth"

acnet = ACCNN(
    size=7,
    in_channels=3,
    num_residual_blocks=8,
    num_filters=32,
    policy_output_dim=49,
    kernel_size=3,
    device="cpu",
)
acnet_onnx = acnet.copy_and_initialize_weights_from_file(filename)
acnet_onnx.initialize_net_from_onnx(acnet_onnx.compile_model_onnx())
# player = ACNetPlayer(acnet_onnx, 'acnet', use_probs_best_2=False)
player = CriticOnlyPlayer(acnet_onnx, 'acnet')

# player = MCTSPlayer("mcts_acnet", acnet_onnx, 400)



# Import and override the `handle_get_action` hook in ActorClient
from oht.ActorClient import ActorClient
class MyClient(ActorClient):
    def handle_game_start(self, start_player):
        """Called at the beginning of of each game
        Args:
        start_player (int): the series_id of the starting player (1 or 2)
        """
        self.logger.info('Game start: start_player=%s', start_player)
        self.total_time_spent = 0
        self.number_of_moves = 0

    def handle_game_over(self, winner, end_state):
        """Called after each game
        Args:
        winner (int): the winning player (1 or 2)
        end_stats (tuple): final board configuration
        Note:
        > Given the following end state for a 5x5 Hex game
        state = [
        2, # Current player is 2 (doesn't matter)
        0, 2, 0, 1, 2, # First row
        0, 2, 1, 0, 0, # Second row
        0, 0, 1, 0, 0, # ...
        2, 2, 1, 0, 0,
        0, 1, 0, 0, 0
        ]
        > Player 1 has won here since there is a continuous
        path of ones from the top to the bottom following the
        neighborhood description given in `handle_get_action`
        """
        self.logger.info('Game over: winner=%s end_state=%s', winner, end_state)

        average_time_per_move = self.total_time_spent / self.number_of_moves
        self.logger.info('Average time per move: %s', average_time_per_move)


    def handle_get_action(self, state):
        current_time = time.time()

        player_turn = state[0]
        player_turn = 1 if player_turn == 1 else -1

        board = np.array(state[1:])
        # replace 2 with -1
        board[board == 2] = -1
        # make board 2d 7x7 instead of 1d 49
        # board = board.reshape(-1, 7)
        # # transpose the board
        # board = board.T

        # rotate board 90 degrees clockwise
        board = board.reshape(7, 7)
        board = np.rot90(board, k=-1)
        
        hex_state = Hex(board, player_turn)
        # print(np.array(list(state[i+1:i+7] for i in range(0, 49, 7))))
        # print(hex_state.tostr())

        i, j = player.get_action(hex_state)
        # rotate back 90 degrees counter clockwise to get row, col
        row, col = int(6-j), int(i)

        self.total_time_spent += time.time() - current_time
        self.number_of_moves += 1

        return row, col

# Initialize and run your overridden client when the script is executed
if __name__ == '__main__':
    client = MyClient(auth="0a3dcdc9d69b425492749713bbb4f4a7", qualify=None)
    client.run(mode="league")
    # client.run()