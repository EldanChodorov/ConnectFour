import pickle
import time
import datetime
import multiprocessing as mp
import queue
import os
import argparse
import sys
import gzip
import subprocess as sp

import numpy as np
import scipy.signal as ss

from policies import base_policy
from policies import *
from initializers import base_initializer
from initializers import *

EMPTY_VAL = 0
PLAYER1_ID = 1
PLAYER2_ID = 2
DRAW = 0
PLAYER1_STRING = "X"
PLAYER2_STRING = "O"
EMPTY_STRING = " "
PLAYERS_STRING = "Player 1: X\nPlayer 2: O\n\n"

ROWS = 6
COLS = 7
WIN_MASK = np.ones(4)
ACTIONS = [0, 1, 2, 3, 4, 5, 6]

WIN_REWARD = 1
LOSE_REWARD = -1
NEUTRAL_REWARD = 0

ILLEGAL_MOVE = "Illegal Move: a random legal move was selected instead. Player tried action"
NO_RESPONSE = "No Response: player took too long to respond with action. This is No Response #"
PLAYER_INIT_TIME = 20
ILLEGAL_BOARD = "Illegal Board: please make sure that the function creates legal boards."
UNRESPONSIVE_PLAYER = "Unresponsive Player: the player hasn't responded in too long... SOMETHING IS WRONG!!"

STATUS_SKIP = 100
SELFPLAY_TIME = 10000
TOO_SLOW_THRESHOLD = 3
UNRESPONSIVE_THRESHOLD = 100


def clear_q(q):
    """
    given a queue, empty it.
    """
    while not q.empty():
        try: q.get_nowait()
        except queue.Empty: break


def days_hours_minutes_seconds(td):
    """
    parse time for logging.
    """
    return td.days, td.seconds//3600, (td.seconds//60)%60, td.seconds%60


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


def render_board(board, game_num, winner, render_rate, last_move=False):
    """
    given a board state, render it on screen.
    :param board: matrix of size (6x7)
    :param game_num: number of game being rendered
    :param winner: who won the game (0 for draw)
    :param render_rate: the rendering rate
    :param last_move: whether this is the last move of the game
    """

    # clear the screen:
    # TODO: change if using linux:
    # print(chr(27)+"[2J") # clear screen for Linux
    os.system('cls')  # clear screen for Windows

    game_id = "Game #" + str(game_num) + "\n"
    board_string = ''
    for row in range(ROWS):
        board_string += "|"
        for col in range(COLS):
            if board[row, col] == PLAYER1_ID:
                board_string += PLAYER1_STRING + "|"
            elif board[row, col] == PLAYER2_ID:
                board_string += PLAYER2_STRING + "|"
            else:
                board_string += EMPTY_STRING + "|"
        board_string += "\n"

    print(game_id + PLAYERS_STRING + board_string)
    if last_move:
        if winner != 0:
            print("Player " + str(winner) + " won the game!")
        else:
            print("The game is a draw!")
    time.sleep(render_rate)


def get_legal_moves(board_state):
    """
    Given a board state, the function returns the legal moves on that board.
    :param board_state: the board state received from the game.
    :return: array of legal moves.
    """

    legal_cols = np.array(np.where(board_state[0, :] == EMPTY_VAL))
    return np.reshape(legal_cols, (legal_cols.size,))


class Board(object):
    """
    a class representing the game board.
    """

    def __init__(self, board_init, game_iteration, starting_player):
        self.game_num = game_iteration
        self.board = board_init.initialize(game_iteration, starting_player)
        assert is_board_legal(self.board)


    def get_board(self):
        """
        :return: the current game state.
        """
        return np.copy(self.board)


    def make_move(self, player_id, col):
        """
        change the game state according to the move the player made. If the
        move chosen was illegal, a random legal move will be chosen.
        :param player_id: the player who acted.
        :param col: the column where the player put a game piece.
        :return: True if the move was legal, False otherwise.
        """

        # check that move is legal and if not, choose random legal move:
        legal_cols = np.array(np.where(self.board[0, :] == EMPTY_VAL))
        legal_cols = np.reshape(legal_cols, (legal_cols.size,))
        if col in legal_cols:
            legality = True
        else:
            legality = False
            col = np.random.choice(legal_cols)

        # update the board:
        row = np.max(np.where(self.board[:, col] == EMPTY_VAL))
        self.board[row, col] = player_id

        return legality


    def check_for_win(self, player_id, col):
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
            if self.board[ROWS-1-i, col] == EMPTY_VAL:
                row = ROWS-i
                break

        # check horizontal:
        vec = self.board[row,:] == player_id
        if np.any(np.convolve(WIN_MASK,vec,mode="valid") == 4):
            return True

        # check vertical:
        vec = self.board[:,col] == player_id
        if np.any(np.convolve(WIN_MASK,vec,mode="valid") == 4):
            return True

        # check diagonals:
        vec = np.diagonal(self.board, col-row) == player_id
        if np.any(np.convolve(WIN_MASK,vec,mode="valid") == 4):
            return True
        vec = np.diagonal(np.fliplr(self.board), COLS-col-1-row) == player_id
        if np.any(np.convolve(WIN_MASK,vec,mode="valid") == 4):
            return True

        return False


class Agent(object):
    SHUTDOWN_TIMEOUT = 15  # seconds until agent is considered unresponsive for model saving

    def __init__(self, id, policy, policy_args, logq, mode, game_duration):
        """
        Construct a new agent
        :param id: the player id
        :param policy: the class of the policy to be used by the player
        :param policy_args: string (name, value) pairs that the policy can parse to arguments
        :param logq: a queue for message logging through the game
        :param mode: whether this is a training or test session
        """
        self.id = id
        self.policy_class = policy
        self.round = 0
        self.unresponsive_count = 0
        self.too_slow = False

        self.sq = mp.Queue()
        self.aq = mp.Queue()
        self.mq = mp.Queue()
        self.logq = logq
        self.policy = policy(policy_args, self.sq, self.aq,
                             self.mq, logq, id, mode, game_duration)
        self.policy.daemon = True
        self.policy.start()


    def handle_state(self, round, prev_state, prev_action, reward, new_state, is_game_over):
        """
        given the new state and previous state-action-reward, pass the information
        to the policy for action selection and/or learning.
        """

        self.round = round
        clear_q(self.sq) # remove previous states from queue if they weren't handled yet
        self.sq.put((round, prev_state, prev_action, reward, new_state, is_game_over, self.too_slow))


    def check_unresponsiveness(self):
        """
        return True if the player hasn't been responsive for too long...
        """
        return self.unresponsive_count > UNRESPONSIVE_THRESHOLD


    def get_action(self, board_state):
        """
        get waiting action from the policy's action queue. if there is no action
        in the queue, pick a random action and log the unresponsiveness error.
        :return: action (column to insert game piece).
        """
        try:
            round, action = self.aq.get_nowait()
            if round != self.round:
                raise queue.Empty()
            else:
                self.too_slow = False
                self.unresponsive_count = 0

        except queue.Empty:
            self.unresponsive_count += 1
            action = np.random.choice(get_legal_moves(board_state))
            if self.unresponsive_count <= UNRESPONSIVE_THRESHOLD:
                self.logq.put((str(self.id), "ERROR", NO_RESPONSE + str(self.unresponsive_count) + " in a row!"))
            if self.unresponsive_count > TOO_SLOW_THRESHOLD:
                self.too_slow = True

        clear_q(self.aq) # clear the queue from unhandled actions
        return action


    def shutdown(self):
        """
        shutdown the agent in the end of the game. the function asks the agent
        to save it's model and returns the saved model, which needs to be a data
        structure that can be pickled.
        :return: the model data structure.
        """
        clear_q(self.sq)
        self.sq.put('save_model')
        try:
            model, save_to = self.mq.get(timeout=Agent.SHUTDOWN_TIMEOUT)
        except queue.Empty:
            model = None  # policy is most probably dead
            save_to = None
            self.logq.put((self.id,'ERROR',"No Model Received From Agent"))
        clear_q(self.aq)
        self.sq.put(None)  # shutdown signal
        self.policy.join()
        return model, save_to


class Game(object):

    @staticmethod
    def log(q, file_name, on_screen=True):
        start_time = datetime.datetime.now()
        logfile = None
        if file_name:
            logfile = gzip.GzipFile(file_name,
                                    'w') if file_name.endswith(
                '.gz') else open(file_name, 'wb')
        for frm, type, msg in iter(q.get, None):
            td = datetime.datetime.now() - start_time
            msg = '%i::%i:%i:%i\t%s\t%s\t%s' % (
            days_hours_minutes_seconds(td) + (frm, type, msg))
            if logfile: logfile.write((msg + '\n').encode('ascii'))
            if on_screen: print(msg)
        if logfile: logfile.close()

    def __init__(self, args):
        self.__dict__.update(args.__dict__)

        self.round = 0

        # check for playback:
        self.is_playback = False
        if self.playback_from is not None:
            self.playback_from = self.record_folder + self.playback_from
            self.is_playback = True
            self.record_to = None

        # initialize logger:
        self.logq = mp.Queue()
        self.logger = mp.Process(target=self.log, args=(
        self.logq, self.log_file))
        self.logger.start()

        # create an archive for recording:
        self.archive = []


    def play_a_game(self, game_num, starting_player):
        """
        play a round of Connect4!
        :param game_num: the round number
        :param starting_player: who acts first
        :return: return the winner's ID (0 for Draws)
        """

        # initialize the board:
        board = Board(self.board_initializer, game_num, starting_player)

        # make sure that the initialized board is legal:
        if not is_board_legal(board.get_board()):
            self.logq.put((str(PLAYER1_ID), "ERROR", ILLEGAL_BOARD))
            return DRAW
        game_record = []
        game_record.append(np.copy(board.get_board()))

        # keep track of previous states:
        states = {}
        states[PLAYER1_ID] = None
        states[PLAYER2_ID] = None
        actions = {}
        actions[PLAYER1_ID] = None
        actions[PLAYER2_ID] = None
        rewards = {}
        rewards[PLAYER1_ID] = NEUTRAL_REWARD
        rewards[PLAYER2_ID] = NEUTRAL_REWARD

        if starting_player == PLAYER1_ID:
            current_player = PLAYER1_ID
            other_player = PLAYER2_ID
        else:
            current_player = PLAYER2_ID
            other_player = PLAYER1_ID

        for _ in range(ROWS*COLS + 1):

            # let the player know about the current state:
            self.players[current_player].handle_state(
                self.round,
                states[current_player],
                actions[current_player],
                rewards[current_player],
                board.get_board(),
                False)
            states[current_player] = board.get_board()

            # get the player action:
            time.sleep(self.policy_action_time)
            actions[current_player] = self.players[current_player].get_action(board.get_board())

            # check if the player is unresponsive for too long:
            if self.players[current_player].check_unresponsiveness():
                self.logq.put((str(current_player), "ERROR", UNRESPONSIVE_PLAYER))

            # make the move and record it:
            previous_board = np.copy(board.get_board())
            legal = board.make_move(current_player, actions[current_player])
            idx = np.nonzero(board.get_board() - previous_board != 0)
            if not legal:
                self.logq.put((str(current_player), "ERROR", ILLEGAL_MOVE + " " + str(actions[current_player])))
                actions[current_player] = idx[1][0]
            game_record.append((idx, board.get_board()[idx]))

            # check if the player won:
            if board.check_for_win(current_player, actions[current_player]):
                rewards[current_player] = WIN_REWARD
                rewards[other_player] = LOSE_REWARD
                self.round += 1

                self.players[current_player].handle_state(
                    self.round,
                    states[current_player],
                    actions[current_player],
                    rewards[current_player],
                    board.get_board(),
                    True)
                self.players[other_player].handle_state(
                    self.round,
                    states[other_player],
                    actions[other_player],
                    rewards[other_player],
                    board.get_board(),
                    True)

                game_record.append(current_player)

                # save the game record:
                if self.record_to is not None and self.playback_from is None:
                    self.archive.append(game_record)

                # update regarding the status:
                if game_num % STATUS_SKIP == 0 and game_num != 0:
                    self.logq.put((0, "STATUS", "Game " + str(game_num) + " over:\n"
                                    "Player 1 wins: " + str(self.score[PLAYER1_ID]) + "\n"
                                    "Player 2 wins: " + str(self.score[PLAYER2_ID]) + "\n"
                                    "Draws: " + str(self.score[DRAW])))

                if self.test_or_train == 'train':
                    time.sleep(self.policy_learn_time)
                return current_player

            # check if it is a draw:
            elif np.all(board.get_board()[0, :] != EMPTY_VAL):
                rewards[current_player] = NEUTRAL_REWARD
                rewards[other_player] = NEUTRAL_REWARD
                self.round += 1

                self.players[current_player].handle_state(
                    self.round,
                    states[current_player],
                    actions[current_player],
                    rewards[current_player],
                    board.get_board(),
                    True)
                self.players[other_player].handle_state(
                    self.round,
                    states[other_player],
                    actions[other_player],
                    rewards[other_player],
                    board.get_board(),
                    True)

                game_record.append(DRAW)

                # save the game record:
                if self.record_to is not None and self.playback_from is None:
                    self.archive.append(game_record)

                # update regarding the status:
                if game_num % STATUS_SKIP == 0 and game_num != 0:
                    self.logq.put((0, "STATUS", "Game " + str(game_num) + " over:\n"
                                    "Player 1 wins: " + str(self.score[PLAYER1_ID]) + "\n"
                                    "Player 2 wins: " + str(self.score[PLAYER2_ID]) + "\n"
                                    "Draws: " + str(self.score[DRAW])))

                if self.test_or_train == 'train':
                    time.sleep(self.policy_learn_time)
                return DRAW


            # otherwise, switch to other player:
            else:
                rewards[current_player] = NEUTRAL_REWARD
                temp = current_player
                current_player = other_player
                other_player = temp


    def run(self):
        """
        The function that runs the entire game
        """
        try:

            # playback the desired games if playback is on:
            if self.is_playback:
                with open(self.playback_from, 'rb') as archive:
                    games = pickle.load(archive)

                if self.initial_playback_round is None:
                    self.initial_playback_round = 0
                if self.final_playback_round is None:
                    self.final_playback_round = len(games)
                if self.playback_round_skip is None:
                    self.playback_round_skip = 1

                for game_num in range(self.initial_playback_round,
                                      self.final_playback_round,
                                      self.playback_round_skip):

                    board = games[game_num][0]
                    moves = games[game_num][1:-1]
                    winner = games[game_num][-1]
                    render_board(board, game_num, winner, self.render_rate, last_move=False)
                    for idx, vals in moves[:-1]:
                        board[idx] = vals
                        render_board(board, game_num, winner, self.render_rate, last_move=False)
                    idx, vals = moves[-1]
                    board[idx] = vals
                    render_board(board, game_num, winner, self.render_rate, last_move=True)

            # otherwise, run the game as requested:
            else:
                game_num = 0
                self.score = {}
                self.score[PLAYER1_ID] = 0
                self.score[PLAYER2_ID] = 0
                self.score[DRAW] = 0

                player_ids = [PLAYER1_ID, PLAYER2_ID]
                starting = 0

                self.players = {}
                self.players[PLAYER1_ID] = Agent(PLAYER1_ID,
                                                 self.agents[0][0],
                                                 self.agents[0][1],
                                                 self.logq,
                                                 self.test_or_train,
                                                 self.game_duration)
                self.players[PLAYER2_ID] = Agent(PLAYER2_ID,
                                                 self.agents[1][0],
                                                 self.agents[1][1],
                                                 self.logq,
                                                 self.test_or_train,
                                                 self.game_duration)

                time.sleep(PLAYER_INIT_TIME)

                while game_num < self.game_duration:

                    # restart the policies every so ofter if we are in selfplay mode:
                    if self.selfplay and game_num % self.selfplay_time == 0 and game_num != 0:

                        # save the player states:
                        player_1_model, player_1_save_to = self.players[PLAYER1_ID].shutdown()
                        player_2_model, player_2_save_to = self.players[PLAYER2_ID].shutdown()
                        if player_1_save_to is not None:
                            player_1_path = player_1_save_to
                            player_1_path = os.path.abspath(self.model_folder) + os.path.sep + player_1_path
                            pickle.dump(player_1_model, open(player_1_path, 'wb'))
                        if player_2_save_to is not None:
                            player_2_path = player_2_save_to
                            player_2_path = os.path.abspath(self.model_folder) + os.path.sep + player_2_path
                            pickle.dump(player_2_model, open(player_2_path, 'wb'))

                        # regenerate the players:
                        self.players[PLAYER1_ID] = Agent(PLAYER1_ID,
                                                         self.agents[0][0],
                                                         self.agents[0][1],
                                                         self.logq,
                                                         self.test_or_train,
                                                         self.game_duration)
                        self.players[PLAYER2_ID] = Agent(PLAYER2_ID,
                                                         self.agents[1][0],
                                                         self.agents[1][1],
                                                         self.logq,
                                                         self.test_or_train,
                                                         self.game_duration)
                        time.sleep(PLAYER_INIT_TIME)

                    winner = self.play_a_game(game_num, player_ids[starting])
                    self.score[winner] += 1
                    game_num += 1
                    starting = 1 - starting

        finally:

            if self.record_to is not None:
                path = os.path.abspath(self.record_folder) + os.path.sep + self.record_to
                with open(path, 'wb') as archive:
                    pickle.dump(self.archive, archive)

            if not self.is_playback:
                print("Out of a total of " + str(self.game_duration) + " games:")
                print("Player 1 wins: " + str(self.score[PLAYER1_ID]))
                print("Player 2 wins: " + str(self.score[PLAYER2_ID]))
                print("Draws: " + str(self.score[DRAW]))

                game_id = str(abs(id(self)))
                num_of_games = self.game_duration

                player_1_string = str(self.players[PLAYER1_ID].policy).split('<')[1].split('(')[0]
                player_2_string = str(self.players[PLAYER2_ID].policy).split('<')[1].split('(')[0]
                player_1_score = self.score[PLAYER1_ID]
                player_2_score = self.score[PLAYER2_ID]

                # save the player states:
                player_1_model, player_1_save_to = self.players[PLAYER1_ID].shutdown()
                player_2_model, player_2_save_to = self.players[PLAYER2_ID].shutdown()
                if player_1_save_to is not None:
                    player_1_path = player_1_save_to
                    player_1_path = os.path.abspath(self.model_folder) + os.path.sep + player_1_path
                    pickle.dump(player_1_model, open(player_1_path, 'wb'))
                else:
                    player_1_path = ''
                if player_2_save_to is not None:
                    player_2_path = player_2_save_to
                    player_2_path = os.path.abspath(self.model_folder) + os.path.sep + player_2_path
                    pickle.dump(player_2_model, open(player_2_path, 'wb'))
                else:
                    player_2_path = ''

                # write the output file:
                output = [','.join(['game_id', 'number_of_games', 'player_num',
                                    'policy', 'number_of_wins', 'model_file_path', 'score'])]
                output.append(','.join([str(game_id), str(num_of_games), str(PLAYER1_ID),
                                         player_1_string, str(player_1_score), player_1_path, str(float(player_1_score)/float(num_of_games))]))
                output.append(','.join([str(game_id), str(num_of_games), str(PLAYER2_ID),
                                         player_2_string, str(player_2_score), player_2_path, str(float(player_2_score)/float(num_of_games))]))
                with open(self.output_file, 'w') as outfile:
                    outfile.write('\n'.join(output))

        # close the logger process:
            self.logq.put(None)
            self.logger.join()


def parse_args():
    p = argparse.ArgumentParser()

    g = p.add_argument_group('I/O')
    g.add_argument('--record_to', '-rt', type=str, default=None, help="file name to which session will be recorded.")
    g.add_argument('--playback_from', '-p', type=str, default=None,
                   help='file name from which game will be played-back (overrides record_to)')
    g.add_argument('--log_file', '-l', type=str, default=None,
                   help="a path to which game events are logged. default: game.log")
    g.add_argument('--output_file', '-o', type=str, default=None,
                   help="a path to a file in which game results and model paths are written. default: game.out")
    g.add_argument('--model_folder', '-mf', type=str, default=None,
                   help="a folder to which agents may record their model'. default: this file's folder, .\states\ ")
    g.add_argument('--record_folder', '-rf', type=str, default=None,
                   help="a folder to which games are recorded'. default: this file's folder, .\\recordings\ ")

    g.add_argument_group('Playback')
    g.add_argument('--initial_playback_round', '-ipr', type=int, default=None,
                   help='which round to start the playback from')
    g.add_argument('--final_playback_round', '-fpr', type=int, default=None,
                   help='which round to end the playback at')
    g.add_argument('--playback_round_skip', '-prs', type=int, default=None,
                   help='how many rounds to skip over between game playbacks')
    g.add_argument('--render_rate', '-rr', type=float, default=1.0,
                   help='seconds to wait between actions when a game is being played back')

    g = p.add_argument_group('Game')
    g.add_argument('--policy_action_time', '-pat', type=float, default=0.01,
                   help='seconds to wait for agents to respond with actions')
    g.add_argument('--policy_learn_time', '-plt', type=float, default=0.1,
                   help='seconds to wait for agents to improve policy')
    g.add_argument('--game_duration', '-D', type=int, default=100, help='number of rounds')
    g.add_argument('--test_or_train', '-t', type=str, default='train',
                   help='whether this is a training session or a test session')
    g.add_argument('--board_initializer', '-bi', type=str, default='RandomBoard',
                   help='function to be used for board initialization')
    g.add_argument('--selfplay', '-sp', type=bool, default=False,
                   help='whether or not this is a selfplay session')
    g.add_argument('--selfplay_time', '-spt', type=int, default=10000,
                   help='whether or not this is a selfplay session')

    g = p.add_argument_group('Agents')
    g.add_argument('--agents', '-A', type=str, default=None,
                   help='a string describing the two agents to be used in the game, of the form: '
                        '<agent_name>(<arg=val>,*);<agent_name>(<arg=val>,*)\n'
                        'e.g. MyAgent(layer1=100,layer2=20);YourAgent(your_params=123)')

    args = p.parse_args()

    # set defaults for I/O:
    code_path = os.path.split(os.path.abspath(__file__))[0] + os.path.sep
    if args.log_file is None:
        args.__dict__['log_file'] = code_path + 'game.log'
    if args.model_folder is None:
        args.__dict__['model_folder'] = code_path + 'models' + os.path.sep
    if not os.path.exists(args.model_folder):
        os.mkdir(args.model_folder)
    if args.record_folder is None:
        args.__dict__['record_folder'] = code_path + 'recordings' + os.path.sep
    if not os.path.exists(args.record_folder):
        os.mkdir(args.record_folder)
    if args.output_file is None:
        args.__dict__['output_file'] = code_path + 'game.out'
    if args.playback_from is not None:
        args.__dict__['record_to'] = None
        args.__dict__['output_file'] = None
        args.__dict__['log_file'] = None
        args.__dict__['model_folder'] = None

    # parse agents:
    agents = []
    if args.agents is not None: agents.extend(args.agents.split(';'))
    if args.playback_from is None: assert len(agents) == 2, "Wrong Number of Players!"

    args.__dict__['agents'] = [base_policy.build(agent) for agent in agents]
    args.__dict__['board_initializer'] = base_initializer.find_initializer(args.__dict__['board_initializer'])
    args.__dict__['board_initializer'] = args.__dict__['board_initializer']()

    return args


if __name__ == '__main__':
    g = Game(parse_args())
    g.run()
