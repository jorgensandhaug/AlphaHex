# class HexMove():

#     def __init__(self, se_diag, ne_diag, board_size) -> None:
#         # assert(se_diag >= 0 and se_diag < board_size and ne_diag >= 0 and ne_diag < board_size)
#         self.se_diag = se_diag
#         self.ne_diag = ne_diag
#         self.board_size = board_size

#     def from_int_representation(int_representation: int, board_size: int) -> HexMove:
#         """ Creates a move from an integer representation 

#         Args:
#             int_representation (int): The integer representation 
#             board_size (int): The board size

#         Returns:
#             HexMove: The corresponding move
#         """
#         return HexMove(int_representation // board_size, int_representation % board_size, board_size)

#     def get_int_representation(self):
#         return self.se_diag * self.board_size + self.ne_diag

#     def as_tuple(self) -> tuple[int, int]:
#         return (self.se_diag, self.ne_diag)

#     def __str__(self) -> str:
#         return f'({self.se_diag}, {self.ne_diag}) -> {self.get_int_representation()}'