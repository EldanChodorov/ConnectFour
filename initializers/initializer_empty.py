from . import base_initializer as bi
import numpy as np


class EmptyBoard(bi.Initializer):
    """
    return an empty board.
    """

    def initialize(self, game_num, starting_player):
        """
        Given a game iteration number, return the board to start the game from.
        :param game_num: game iteration number
        :param starting_player: player that will start the game
        :return: board - numpy array of size (6,7)
        """

        return np.zeros((6,7))