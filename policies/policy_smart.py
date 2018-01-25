import numpy as np
from policies.base_policy import Policy
import random
import time
from collections import deque
import tensorflow as tf
import pickle


EMPTY_VAL = 0
ROWS = 6
COLS = 7
WIN_MASK = np.ones(4)
ACTIONS = [0, 1, 2, 3, 4, 5, 6]
NUM_ACTION = 7


class TransitionBatch:
    '''TransitionBatch represents <state,action,rewards,...> seen in a single turn.'''
    def __init__(self, state, action, reward, next_state, prev_win_vector, new_winning_vec):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.prev_winning_vec = prev_win_vector
        self.new_winning_vec = new_winning_vec


class Policy200039626(Policy):

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
        '''
        Extract from give q_values the ones which are valid according to the given states.
        :return: vector of valid q values.
        '''
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
        '''Perform learning process of the policy.'''

        try:
            # update values of epsilon for exploration-exploitation tradeoff
            if self.mode == 'train' and self.next_decay == round:
                if self.epsilon > 0.05:
                    self.epsilon = self.epsilon* (0.8)
                    self.next_decay += 2 * self.epsilon_decay_round
                if np.abs(self.game_duration/2 - self.next_decay) < 5:
                    self.epsilon = 0.05
                    self.next_decay = 0

            # if lean is taking too long, learn less batches each time
            if too_slow:
                if self.batch_size > 30:
                    self.batch_size -= 5
            total_time_past = 0

            while total_time_past + self.norm_learn_time < self.policy_learn_time:
                if len(self.round_time_list) >= 1000:
                    self.round_time_list.popleft()
                start_time = time.time()

                # if reward is -1/1 then it was not passed to act() and must be stored in memory
                if reward != 0:

                    if len(self.transitions_memory) >= self.memory_limit:
                        self.transitions_memory.popleft()

                    # normalize board
                    encapsulated_new_state = self.hot_boards(new_state)
                    new_winning_vec = self.get_winning_vector_with_enemies(new_state, self.id, self.enemy_id)

                    # not the first turn
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
                feed_dict = {self.rewards: fixed_rewards, self.actions_holder: vector_actions,
                             self.boards: states ,self.winning_vec: prev_winning_vec}
                _, loss, net = self.session.run([self.optimizer, self.loss, self.q_vals], feed_dict=feed_dict)
                round_tim = time.time() - start_time
                total_time_past += round_tim
                self.round_time_list.append(round_tim)

            #update mean_learn time
            self.norm_learn_time = np.mean(np.array(self.round_time_list)) + np.std(np.array(self.round_time_list))/4

        except Exception as ex:
            print("Exception in learn: %s %s" % (type(ex), ex))

    def get_next_Q(self, curr_state, winning_vec):
        return self.q_vals.eval(feed_dict={self.boards: curr_state, self.winning_vec: winning_vec},
                                session=self.session)

    def build_next_state(self, state, action, player_id):
        '''Construct matrix representing the next state after the given state and action.'''
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

    def normalize_board(self, board):
        '''
        Normalize board to values -1 (enemy) 1 (myself) 0 (else)
        :param board:
        :return: normalized board
        '''
        board = np.copy(board)
        if self.id == 1:
            board[np.where(board == 2)] = -1
        else:
            board[np.where(board == 1)] = -1
            board[np.where(board == 2)] = 1
        return board

    def get_winning_vector_helper(self, state, player_id):
        '''Produce one hot vector with 1 for an action which will give a winning move.'''
        vec = np.zeros((1, NUM_ACTION, 1))
        valid_columns = np.where(state[0, :] == 0)[0]
        for i in valid_columns:
            simulated_state = self.build_next_state(state, i, player_id)
            if self.check_for_win(simulated_state, player_id, i):
                vec[0, i, 0] = 1
                break
        return vec

    def get_winning_vector_with_enemies(self, state, player1_id, player2_id):
        '''
        Create winning vectors one-hot vectors.
        :return: concatenation of 2 winning vectors, one for self and one for enemy
        '''
        my_winning = self.get_winning_vector_helper(state, player1_id)
        enemy_winning = self.get_winning_vector_helper(state, player2_id)
        together = np.concatenate((my_winning, enemy_winning))
        return together.reshape((1, 7, 2))

    def single_hot_board(self, board, player_id):
        ''':return the board as a binary matrix only for the player_id.'''
        state = np.zeros(board.shape)
        state[np.where(board == player_id)] = 1
        return state[..., None]

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        '''Perform round of act - choose and return action.'''

        try:
            encapsulated_boards = self.hot_boards(new_state)
            winning_vec = self.get_winning_vector_with_enemies(new_state, self.id, self.enemy_id)

            # use epsilon greedy
            if np.random.rand() < self.epsilon:

                # choose random action
                action = self.get_random_action(new_state)

            else:
                # get next action from network
                action = self.get_qNet_action(encapsulated_boards, winning_vec)

            # store parameters in memory
            if self.mode == 'train':
                if prev_action is not None and prev_state is not None:
                    prev_encapsulated_boards = self.hot_boards(prev_state)
                    prev_winning_vec = self.get_winning_vector_with_enemies(prev_state, self.id, self.enemy_id)

                    self.transitions_memory.append(TransitionBatch(prev_encapsulated_boards, prev_action, reward,
                                                                   encapsulated_boards, prev_winning_vec, winning_vec))

                    # clear memory if full
                    if len(self.transitions_memory) >= self.memory_limit:
                        self.transitions_memory.popleft()

        except Exception as ex:
            print("Exception in act: %s %s" %(type(ex), ex))
            action = np.random.choice(np.arange(NUM_ACTION))

        finally:
            return action

    def hot_boards(self, board):
        ''':return binary matrices representing boards of each player and enemy.'''
        my_board = self.single_hot_board(board, self.id)
        enemy_board = self.single_hot_board(board, self.enemy_id)
        return np.concatenate((my_board, enemy_board), axis=-1)

    def get_random_action(self, new_state):
        action = np.random.choice(np.arange(NUM_ACTION))

        # avoid illegal moves
        invalid = new_state[0, action] != 0
        while invalid:
            action = np.random.choice(np.arange(NUM_ACTION))
            invalid = new_state[0, action] != 0
        return action

    def get_qNet_action(self, new_state, winning_vec):
        ''':return action chosen by policy'''
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
            w = self.session.run(v)
            weights.append(w)
        print("Model saved to %s" % self.save_to)
        return weights, self.save_to

    def network_run(self, board, action_vec):
        '''
        Neural network with architecture:
        
        convolution layer (boards only)
        activation tanh

        convolution layer (boards only)
        activation tanh
    
        convolution layer (boards only)
        activation tanh
        
        convolution layer (boards only)
        activation tanh

        fully connected layer (boards & vector)
        activation tanh
        
        :param board: binary matrices of player's circles and of enemy's circles
        :param action_vec: one-hot vector of length 7
        :return: vector of length 7 (Q values)
        '''
        conv_layer1 = tf.contrib.layers.conv2d(board, 32, [5, 5], padding='SAME', activation_fn=tf.nn.sigmoid)
        conv_layer2 = tf.contrib.layers.conv2d(conv_layer1, 16, [5, 5], padding='SAME', activation_fn=tf.nn.sigmoid)
        conv_layer3 = tf.contrib.layers.conv2d(conv_layer2, 16, [5, 5], padding='SAME', activation_fn=tf.nn.sigmoid)
        conv_layer4 = tf.contrib.layers.conv2d(conv_layer3, 1, [5, 5], padding='SAME', activation_fn=tf.nn.sigmoid)

        out_shape = conv_layer4.get_shape().as_list()
        flat_out = tf.reshape(conv_layer4, [-1, out_shape[1] * out_shape[2]])
        fc = tf.contrib.layers.fully_connected(flat_out, 14, tf.nn.tanh,
                                               weights_initializer=tf.random_normal_initializer(),
                                               biases_initializer=tf.random_normal_initializer())
        fc = tf.reshape(fc, [-1, 14])
        action_vec = tf.reshape(action_vec, [-1, 14])

        together = tf.concat(1, [fc, action_vec])
        connect_fc = tf.contrib.layers.fully_connected(together, 7, biases_initializer=tf.random_normal_initializer(),
                                                       scope='fc6', weights_initializer=tf.random_normal_initializer(),
                                                       activation_fn=tf.nn.tanh)
        final = tf.reshape(connect_fc, [-1, 7, 1])
        return final

    def init_network(self):
        '''Initialize neural network.'''

        self.actions_holder = tf.placeholder(tf.float32, shape=[None, 7, 1], name="action_holder")
        self.boards = tf.placeholder(tf.float32, shape=[None, 6, 7, 2], name="boards")
        self.winning_vec = tf.placeholder(tf.float32, shape=[None, 7, 2], name="actions_vector")
        self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")

        self.q_vals = self.network_run(self.boards, self.winning_vec)
        self.action = tf.reduce_max(self.q_vals * self.actions_holder, reduction_indices=1)
        self.loss = tf.reduce_mean(tf.pow(self.rewards - self.action, 2))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.init = tf.initialize_all_variables()
        self.session = tf.Session()
        with self.session.as_default():
            self.session.run(self.init)

    def init_run(self):
        # store all transition batches seen during game {round_num: transition_batch}
        self.transitions_memory = deque()
        self.norm_learn_time = 0.01
        self.round_time_list = deque()

        # do not explore during test time
        if self.mode == 'test':
            self.epsilon = 0

        # initialize neural network
        self.init_network()

        # load stored model
        self.epsilon_decay_round = int(self.game_duration / 100)
        self.next_decay = self.epsilon_decay_round

        # load model
        if self.load_from:
            load_path = self.load_from
        else:
            load_path = 'models/' + self.save_to
        try:
            with open(load_path, 'rb') as f:
                weights = pickle.load(f)
                for v, w in zip(tf.trainable_variables(), weights):
                    self.session.run(v.assign(w))
            print("Model loaded from %s" % self.load_from)
        except FileNotFoundError:
            # first run, no model to load
            pass

        # set enemy id
        if self.id == 1:
            self.enemy_id = 2
        else:
            self.enemy_id = 1
