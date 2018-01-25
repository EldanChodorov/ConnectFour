import numpy as np
from policies.base_policy import Policy
from policies.topologies import PolicyNetwork
import random
import time
from collections import deque
import tensorflow as tf
import pickle
from scipy import signal
from scipy.ndimage import morphology as m

EMPTY_VAL = 0
ROWS = 6
COLS = 7
WIN_MASK = np.ones(4)
ACTIONS = [0, 1, 2, 3, 4, 5, 6]

DEBUG = True


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.4f' % (method.__name__, (te - ts)))
        return result
    return timed


class TransitionBatch:
    def __init__(self, state, action, reward, next_state, prev_win_vector, new_winning_vec):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.prev_winning_vec = prev_win_vector
        self.new_winning_vec = new_winning_vec


NUM_ACTION = 7


class SmartPolicy(Policy):

    def cast_string_args(self, policy_args):
        policy_args['batch_size'] = int(policy_args['batch_size']) if 'batch_size' in policy_args else 40
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else 0.95
        policy_args['gamma'] = float(policy_args['gamma']) if 'gamma' in policy_args else 0.99
        policy_args['learning_rate'] = float(policy_args['learning_rate']) if 'learning_rate' in policy_args else 0.0001
        policy_args['learning_decay'] = float(policy_args['learning_decay']) if 'learning_decay' in policy_args else 0
        policy_args['epsilon_decay'] = float(policy_args['epsilon_decay']) if 'epsilon_decay' in policy_args else 0.1
        policy_args['memory_limit'] = int(policy_args['memory_limit']) if 'memory_limit' in policy_args else 30000
        policy_args['save_to'] = policy_args['save_to'] if 'save_to' in policy_args else None
        policy_args['load_from'] = policy_args['load_from'] if 'load_from' in policy_args else None
        policy_args['c_iters'] = policy_args['c_iters'] if 'c_iters' in policy_args else 100
        policy_args['policy_learn_time'] = policy_args['policy_learn_time'] if 'policy_learn_time' in policy_args else 0.1
        self.log(policy_args)
        return policy_args

    def get_random_batches(self, batch_size):
        # select random batches from memory
        random_batches = random.sample(self.transitions_memory, batch_size)

        # extract features from batches
        rewards = np.array([batch.reward for batch in random_batches])
        states = np.array([batch.state for batch in random_batches])
        next_states = np.array([batch.next_state for batch in random_batches])
        actions = np.array([batch.action for batch in random_batches])
        prev_winning_vecs = np.array([batch.prev_winning_vec for batch in random_batches])
        new_winning_vecs = np.array([batch.new_winning_vec for batch in random_batches])

        # reshape
        vector_actions = (np.eye(7)[actions]).reshape((batch_size, 7, 1))

        next_states = next_states.reshape((batch_size, 6, 7, 2))
        states = states.reshape((batch_size, 6, 7, 2))
        prev_winning_vecs = prev_winning_vecs.reshape((batch_size, 7, 2))
        new_winning_vecs = new_winning_vecs.reshape((batch_size, 7, 2))
        return rewards, states, next_states, vector_actions, prev_winning_vecs, new_winning_vecs

    def get_valid_Qvals(self, states, q_values, batch_size):

        action_table = np.fliplr(np.argsort(np.squeeze(q_values, axis=-1), axis=1))
        best_q = np.zeros(batch_size)

        # replace invalid moves
        for single_batch in np.arange(batch_size):
            single_example = action_table[single_batch]
            for i, action in enumerate(single_example):
                if states[single_batch, 0, action, 0] == 0 and states[single_batch, 0, action, 1] == 0:
                    best_q[single_batch] = q_values[single_batch, action, 0]
                    break
        return best_q

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        try:
            if self.mode == 'train' and self.next_decay == round:
                if self.epsilon > 0.05:
                    self.epsilon  = self.epsilon* (0.8)
                    self.next_decay += 2 * self.epsilon_decay_round
                if np.abs(self.game_duration/2 - self.next_decay) < 5:
                    self.epsilon = 0.05
                    self.next_decay = 0

            if too_slow:
                if self.batch_size > 30:
                    self.batch_size -= 5
            totel_time_past = 0

            while totel_time_past + self.norm_learn_time < self.policy_learn_time:
                if len(self.round_time_list) >= 1000:
                    self.round_time_list.popleft()
                start_time = time.time()
                if reward != 0:
                    if DEBUG:
                        if not self._round_started and round % 500 == 0:
                            self._round_started = True

                        if self._round_started and round % 500 != 0:
                            self._round_started = False

                    if len(self.transitions_memory) >= self.memory_limit:
                        self.transitions_memory.popleft()

                    # normalize board
                    encapsulated_new_state = self.hot_boards(new_state)
                    new_winning_vec = self.get_winning_vector_with_enemies(new_state, self.id, self.enemy_id)

                    if prev_action is not None and prev_state is not None:
                        encapsulated_prev_state = self.hot_boards(prev_state)

                        # store parameters in memory
                        prev_winning_vec = self.get_winning_vector_with_enemies(prev_state, self.id, self.enemy_id)
                        self.transitions_memory.append(TransitionBatch(encapsulated_prev_state, prev_action, reward,
                                                                       encapsulated_new_state, prev_winning_vec, new_winning_vec))

                # set batch size
                if self.batch_size < len(self.transitions_memory):
                    batch_size = self.batch_size
                else:
                    batch_size = len(self.transitions_memory)

                # select random batches from memory
                rewards, states, next_states, vector_actions, prev_winning_vec, new_winning_vec = self.get_random_batches(batch_size)

                # get next action from network and filter valid moves
                new_q = self.get_next_Q(next_states, new_winning_vec)
                best_q = self.get_valid_Qvals(next_states, new_q, batch_size)

                fixed_rewards = np.clip(rewards + (self.gamma * best_q) * (1 - np.square(rewards)), -1, 1)


                # train
                feed_dict = {self.q_network.rewards: fixed_rewards, self.q_network.actions_holder: vector_actions,
                             self.q_network.boards: states ,self.q_network.winning_vec: prev_winning_vec}
                _, loss, net = self.q_network.session.run([self.q_network.optimizer, self.q_network.loss,
                                                           self.q_network.q_vals], feed_dict=feed_dict)
                round_tim = time.time() - start_time
                totel_time_past += round_tim
                self.round_time_list.append(round_tim)
            #updaet mean_learn time
            self.norm_learn_time = np.mean(np.array(self.round_time_list)) + np.std(np.array(self.round_time_list))/4
            print(self.norm_learn_time)
            if DEBUG and round % 100 == 0:
                self.log("ROUND {} loss {}".format(round, loss))

        except Exception as ex:
            print("Exception in learn: %s %s" % (type(ex), ex))

    def get_next_Q(self, curr_state, winning_vec):
        return self.q_network.q_vals.eval(feed_dict={self.q_network.boards: curr_state, self.q_network.winning_vec: winning_vec},
                                          session=self.q_network.session)

    def build_next_state(self, state, action, player_id):
        if state[0, action] != 0:
            return state
        state = np.copy(state)
        row = np.max(np.where(state[:, action] == 0))
        state[row, action] = player_id
        return state

    def generate_sample_final_board(self, state, action):
        # generate best next action state, in order to feed next state into network
        row = np.max(np.where(state[:, action] == 0))
        next_board = np.copy(state)
        next_board[row, action] = 1
        return next_board

    def update_rates(self):
        # learning rate
        if self.q_network.lr > 0.005:
            self.q_network.lr -= self.learning_decay
        # exploration-exploitation e-greedy
        if self.epsilon > 0.2:
            self.epsilon -= self.epsilon_decay

    def check_for_win(self, board, player_id, col):
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

    def get_me_edges(self,state,mask):
        convolv_stat = signal.convolve2d(state, mask, mode="same")
        convolv_stat[convolv_stat != 3] = 0
        line_in_state = m.binary_dilation(convolv_stat.astype(np.bool),
                                                   structure=mask).astype(np.float32)
        edges_in_state = (m.binary_dilation(line_in_state.astype(np.bool),
                                                   structure=mask).astype(np.float32) - line_in_state).astype(np.float32)
        return edges_in_state

    def re_represent_state(self, state):
        """

        :param board: the board state.
        :return: True iff board is legal.
        """

        our_palce = state[:,:] == 1
        us_three_in_a_row_mask = self.get_me_edges(our_palce,self.WIN_MASK)
        us_three_in_a_line_mask  = self.get_me_edges(our_palce,self.WIN_MASK.T)
        us_three_in_a_diag1_mask = self.get_me_edges(our_palce, np.identity(3))
        us_three_in_a_diag2_mask = self.get_me_edges(our_palce, np.flip(np.identity(3),axis=1))

        here_place = state[:, :] == -1
        here_three_in_a_row_mask = self.get_me_edges(here_place, self.WIN_MASK)
        here_three_in_a_line_mask = self.get_me_edges(here_place, self.WIN_MASK.T)
        here_three_in_a_diag1_mask = self.get_me_edges(here_place, np.identity(3))
        here_three_in_a_diag2_mask = self.get_me_edges(here_place, np.flip(np.identity(3),axis=1))

        rows = ((us_three_in_a_row_mask + here_three_in_a_row_mask))[None,:,:]
        lines = ((us_three_in_a_line_mask + here_three_in_a_line_mask))[None,:,:]
        diag1 = ((here_three_in_a_diag1_mask + us_three_in_a_diag1_mask))[None,:,:]
        diag2 = ((here_three_in_a_diag2_mask + us_three_in_a_diag2_mask))[None,:,:]
        state = state[None,:,:]
        full_state = np.concatenate((state,rows,lines,diag1,diag2),axis=0)
        # for stat in full_state:
        #     plt.imshow(stat,cmap = 'gray')
        #     plt.show()
        return full_state

    def normalize_board(self, board):
        '''
        Normalize board to values -1 (enemy) 1 (myself) 0 (else)
        :param board:
        :return:
        '''
        board = np.copy(board)
        if self.id == 1:
            board[np.where(board == 2)] = -1
        else:
            board[np.where(board == 1)] = -1
            board[np.where(board == 2)] = 1
        return board

    def get_winning_vector_helper(self, state, player_id):
        vec = np.zeros((1, NUM_ACTION, 1))
        valid_columns = np.where(state[0, :] == 0)[0]
        for i in valid_columns:
            simulated_state = self.build_next_state(state, i, player_id)
            if self.check_for_win(simulated_state, player_id, i):
                vec[0, i, 0] = 1
                break
        return vec

    def get_winning_vector_with_enemies(self, state, player1_id, player2_id):
        my_winning = self.get_winning_vector_helper(state, player1_id)
        enemy_winning = self.get_winning_vector_helper(state, player2_id)
        together = np.concatenate((my_winning, enemy_winning))
        return together.reshape((1, 7, 2))

    def get_winning_vector(self, state1, state2, player1_id, player2_id):
        winning1 = self.get_winning_vector_helper(state1, player1_id)
        winning2 = self.get_winning_vector_helper(state2, player2_id)
        together = np.concatenate((winning1, winning2))
        return together

    def single_hot_board(self, board, player_id):
        state = np.zeros(board.shape)
        state[np.where(board == player_id)] = 1
        return state[..., None]

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        # log game state
        if DEBUG:
            if self._round_started:
                if reward != 0:
                    self._round_started = False

        try:

            encapsulated_boards = self.hot_boards(new_state)
            winning_vec = self.get_winning_vector_with_enemies(new_state, self.id, self.enemy_id)

            # use epsilon greedy
            if np.random.rand() < self.epsilon:
                if DEBUG and self.mode == 'test':
                    print("getting RANDOM action")

                # choose random action
                action = self.get_random_action(new_state)
                self.num_random_moves_taken += 1

            else:
                # get next action from network
                self.num_q_moves_takin += 1
                action = self.get_qNet_action(encapsulated_boards, winning_vec)

            if self.mode == 'train':
                if prev_action is not None and prev_state is not None:
                    prev_encapsulated_boards = self.hot_boards(prev_state)
                    prev_winning_vec = self.get_winning_vector_with_enemies(prev_state, self.id, self.enemy_id)
                    # prev_winning_vec = self.get_winning_vector_helper(prev_state, self.id)
                    # print(prev_encapsulated_boards,'\n')
                    # print(prev_winning_vec,'\n')
                    # print(prev_action, '\n')
                    # store parameters in memory
                    self.transitions_memory.append(TransitionBatch(prev_encapsulated_boards, prev_action, reward,
                                                                   encapsulated_boards, prev_winning_vec, winning_vec))

                    # clear memory if full
                    if len(self.transitions_memory) >= self.memory_limit:
                        self.transitions_memory.popleft()

                    # update learning rate
                    if round % 1000 == 0:
                        self.update_rates()

        except Exception as ex:
            print("Exception in act: %s %s" %(type(ex), ex))
            action = np.random.choice(np.arange(NUM_ACTION))

        finally:
            # print("Player %d: Action %d" %(self.id, action))
            # print("Board from player %d with chosen action %d" % (self.id, action))
            # print(new_state)
            return action

    def hot_boards(self, board):
        my_board = self.single_hot_board(board, self.id)
        enemy_board = self.single_hot_board(board, self.enemy_id)
        return np.concatenate((my_board, enemy_board), axis=-1)

    def get_random_action(self, new_state):
        action = np.random.choice(np.arange(NUM_ACTION))

        # avoid illegal moves
        invalid = new_state[0, action] != 0
        while invalid:
            action = np.random.choice(np.arange(NUM_ACTION))  # TODO without illegal actions
            invalid = new_state[0, action] != 0
        return action

    def get_qNet_action(self, new_state, winning_vec):
        new_state = new_state.reshape((1, 6, 7, 2))
        q_values = self.get_next_Q(new_state, winning_vec)
        action_table = np.fliplr(np.argsort(q_values, axis=1))

        # choose first valid action
        chosen_action = 0
        for action in action_table[0, :, 0]:
            if new_state[0, 0, action, 0] == 0 and new_state[0, 0, action, 1] == 0:
                chosen_action = action
                break

        return chosen_action

    def save_model(self):
        weights = []
        for v in tf.trainable_variables():
            w = self.q_network.session.run(v)
            weights.append(w)
        print("Model saved to %s" % self.save_to)
        print("%d random moves and %d Q moves" % (self.num_random_moves_taken, self.num_q_moves_takin))
        return weights, self.save_to

    def init_run(self):
        # store all transition batches seen during game {round_num: transition_batch}
        self.transitions_memory = deque()
        self.norm_learn_time = 0.01
        self.round_time_list = deque()
        print('MODE: %s' % self.mode)

        if self.mode == 'test':
            self.epsilon = 0



        self.q_network = PolicyNetwork(self.learning_rate, epochs=5, batches_per_epoch=self.batch_size)

        self.log("SmartPlayer id: {}".format(self.id))

        # for debugging
        self._round_started = False

        # load stored model
        self.WIN_MASK = np.ones(3)[..., None]
        self.ROWS = 6
        self.COLS = 7
        self.epsilon_decay_round = int(self.game_duration / 100)
        self.next_decay = self.epsilon_decay_round
        if self.load_from:
            load_path = self.load_from
        else:
            load_path = 'models/' + self.save_to
        try:
            with open(load_path, 'rb') as f:
                weights = pickle.load(f)
                for v, w in zip(tf.trainable_variables(), weights):
                    self.q_network.session.run(v.assign(w))
            print("Model loaded from %s" % self.load_from)
        except FileNotFoundError:
            # first run, no model to load
            pass

        if self.id == 1:
            self.enemy_id = 2
        else:
            self.enemy_id = 1

        self.num_random_moves_taken = 0
        self.num_q_moves_takin = 0