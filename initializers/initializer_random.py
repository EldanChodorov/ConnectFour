from . import base_initializer as bi
import numpy as np

MAX_PIECES = 10
EMPTY_VAL = 0
PLAYER1_ID = 1
PLAYER2_ID = 2
ROWS = 6
COLS = 7
WIN_MASK = np.ones(4)

# TODO: problem with giving the players a board where 1 move wins - someone needs to learn with nothing in the history (throws an error)

def is_board_legal(board):
    """
    given an initial board, check that it is legal (no one has won yet and the size is OK).
    :param board: the board state.
    :return: True iff board is legal.
    """

    # check that the board size is legal:
    rows, cols = np.shape(board)
    if rows != ROWS or cols != COLS:
        return False

    # check horizontals:
    for row in range(ROWS):
        vec = board[row, :] == PLAYER1_ID
        if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
            return False
        vec = board[row, :] == PLAYER2_ID
        if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
            return False

    # check verticals:
    for col in range(COLS):
        vec = board[:, col] == PLAYER1_ID
        if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
            return False
        vec = board[:, col] == PLAYER2_ID
        if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
            return False

    # check diagonals:
    for diag in range(ROWS + COLS - 1):
        vec = np.diagonal(board, diag - ROWS + 1) == PLAYER1_ID
        if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
            return False
        vec = np.diagonal(np.fliplr(board), diag - ROWS + 1) == PLAYER1_ID
        if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
            return False
        vec = np.diagonal(board, diag - ROWS + 1) == PLAYER2_ID
        if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
            return False
        vec = np.diagonal(np.fliplr(board), diag - ROWS + 1) == PLAYER2_ID
        if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
            return False

    return True


class RandomBoard(bi.Initializer):
    """
    return a legal, random board with up to MAX_PIECES game pieces.
    """

    def initialize(self, game_num, starting_player):
        """
        Given a game iteration number, return the board to start the game from.
        :param game_num: game iteration number
        :param starting_player: player that will start the game
        :return: board - np.matrix of size (6,7)
        """

        starting_player -= 1
        players = [PLAYER1_ID, PLAYER2_ID]

        while True:

            board = np.zeros((6, 7))

            # get the number of pieces to insert:
            num_pieces = np.random.randint(MAX_PIECES)
            if num_pieces % 2 == 1:
                num_pieces += 1

            # insert the pieces one by one:
            for _ in range(num_pieces):

                legal_cols = np.array(np.where(board[0, :] == EMPTY_VAL))
                legal_cols = np.reshape(legal_cols, (legal_cols.size,))
                col = np.random.choice(legal_cols)
                row = np.max(np.where(board[:, col] == EMPTY_VAL))
                board[row, col] = players[starting_player]
                starting_player = 1 - starting_player

            # make sure board is legal:
            if is_board_legal(board):
                break

        return board