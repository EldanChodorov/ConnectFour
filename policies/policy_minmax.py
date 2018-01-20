from policies import base_policy as bp
import numpy as np
import tensorflow as tf
import pickle

EMPTY_VAL = 0
PLAYER1_ID = 1
PLAYER2_ID = 2

ROWS = 6
COLS = 7
WIN_MASK = np.ones(4)
ACTIONS = [0, 1, 2, 3, 4, 5, 6]

def check_for_win(board, player_id, col):
    """
    check the board to see if last move was a winning move.
    :param board: the new board
    :param player_id: the player who made the move
    :param col: his action
    :return: True iff the player won with his move
    """

    row = 0

    # check which row was inserted last:
    for i in range(ROWS):
        if board[ROWS - 1 - i, col] == EMPTY_VAL:
            row = ROWS - i
            break

    # check horizontal:
    vec = board[row, :] == player_id
    if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
        return True

    # check vertical:
    vec = board[:, col] == player_id
    if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
        return True

    # check diagonals:
    vec = np.diagonal(board, col - row) == player_id
    if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
        return True
    vec = np.diagonal(np.fliplr(board), COLS - col - 1 - row) == player_id
    if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
        return True

    return False


def make_move(board, action, player_id):
    """
    return a new board with after performing the given move.
    :param board: original board
    :param action: move to make (column)
    :param player_id: player that made the move
    :return: new board after move was made
    """
    row = np.max(np.where(board[:, action] == EMPTY_VAL))
    new_board = np.copy(board)
    new_board[row, action] = player_id

    return new_board


def minmax(board, depth=1, to_max=True, player_id=PLAYER1_ID):
    """
    The Minmax recursive function, calculating the optimal move for the player
    up to given depth.
    :param board: the board state
    :param depth: how deep to look in the game tree
    :param to_max: whether this is the player's turn or other player's turn (to_max
                    being True means that it is the player's turn)
    :param player_id: the ID of the player who's turn it is (for understanding the board)
    :return: (best_action, expected_score_from_action)
    """

    other_player_id = PLAYER1_ID
    if player_id == PLAYER1_ID:
        other_player_id = PLAYER2_ID

    # find legal actions:
    legal_actions = np.array(np.where(board[0, :] == EMPTY_VAL))
    legal_actions = np.reshape(legal_actions, (legal_actions.size,))


    # in case of weird problems and draws (no legal actions):
    if len(legal_actions) == 0:
        return 0, 0

    # stopping case for recursion:
    if depth == 0:
        return np.random.choice(legal_actions), 0

    # if we want to maximize the result:
    if to_max:
        expected_scores = -2 * np.ones(len(ACTIONS))
        for action in legal_actions:
            new_board = make_move(board, action, player_id)
            if check_for_win(new_board, player_id, action):
                return action, 1
            _, expected_scores[action] = minmax(new_board, depth-1, not to_max, other_player_id)
        return np.random.choice(np.flatnonzero(expected_scores == np.max(expected_scores))), np.max(expected_scores)

    # if we want to minimize:
    else:
        expected_scores = 2 * np.ones(len(ACTIONS))
        for action in legal_actions:
            new_board = make_move(board, action, player_id)
            if check_for_win(new_board, player_id, action):
                return action, -1
            _, expected_scores[action] = minmax(new_board, depth-1, not to_max, other_player_id)
        return np.random.choice(np.flatnonzero(expected_scores == np.min(expected_scores))), np.min(expected_scores)


class MinmaxAgent(bp.Policy):
    """
    An agent performing the Minmax Algorithm for a given depth. The agent will
    return the right moves if there is a win or a correct defensive move for
    the given depth, and otherwise act randomly.
    """

    def cast_string_args(self, policy_args):

        policy_args['depth'] = int(policy_args['depth']) if 'depth' in policy_args else 1
        return policy_args


    def init_run(self):

        # example of loading a pickled Tensorflow model:
        try:
            model = pickle.load(open(self.load_from, 'rb'))

            self.W = tf.Variable(tf.constant(model[0]))
            self.b = tf.Variable(tf.constant(model[1]))

        except:

            self.W = tf.truncated_normal([ROWS*COLS*2, 7], stddev=0.1, dtype=tf.float32)
            self.b = tf.constant(0, shape=[7], dtype=tf.float32)

            self.log("Model not found, initializing random weights.", 'STATUS')

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        return


    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        return


    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        legal_actions = np.array(np.where(new_state[0, :] == EMPTY_VAL))
        legal_actions = np.reshape(legal_actions, (legal_actions.size,))

        action, score = minmax(new_state, self.depth, True, self.id)

        if action in legal_actions:
            return action
        else:
            return np.random.choice(legal_actions)


    def save_model(self):

        return [self.session.run(self.W), self.session.run(self.b)], None
