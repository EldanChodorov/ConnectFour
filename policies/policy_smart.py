import numpy as np
from policies.base_policy import Policy
from policies.topologies import PolicyNetwork
import random
import time
from collections import deque
import tensorflow as tf
import pickle

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
            print ('%r  %2.4f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

class TransitionBatch:
    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state


NUM_ACTION = 7


class SmartPolicy(Policy):

    def cast_string_args(self, policy_args):
        policy_args['batch_size'] = int(policy_args['batch_size']) if 'batch_size' in policy_args else 50
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else 0.1
        policy_args['gamma'] = float(policy_args['gamma']) if 'gamma' in policy_args else 0.95
        policy_args['learning_rate'] = float(policy_args['learning_rate']) if 'learning_rate' in policy_args else 0.1
        policy_args['learning_decay'] = float(policy_args['learning_decay']) if 'learning_decay' in policy_args else 0.005
        policy_args['epsilon_decay'] = float(policy_args['epsilon_decay']) if 'epsilon_decay' in policy_args else 0.001
        policy_args['memory_limit'] = int(policy_args['memory_limit']) if 'memory_limit' in policy_args else 5000
        policy_args['save_to'] = policy_args['save_to'] if 'save_to' in policy_args else None
        policy_args['load_from'] = policy_args['load_from'] if 'load_from' in policy_args else None
        return policy_args

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        try:

            if reward != 0:

                if DEBUG:
                    if not self._round_started and round % 500 == 0:
                        self._round_started = True

                if len(self.transitions_memory) >= self.memory_limit:
                    self.transitions_memory.popleft()

                # normalize board
                new_state = self.normalize_board(new_state)

                if prev_action is not None and prev_state is not None:
                    prev_state = self.normalize_board(prev_state)

                    # store parameters in memory
                    self.transitions_memory.append(TransitionBatch(prev_state, prev_action, reward, new_state))

            # set batch size
            if self.batch_size < len(self.transitions_memory):
                batch_size = self.batch_size
            else:
                batch_size = int(self.batch_size / 2)
            if batch_size > len(self.transitions_memory):
                return

            # select random batches from memory
            random_batches = random.sample(self.transitions_memory, batch_size)

            # extract features from batches
            rewards = np.array([batch.reward for batch in random_batches])
            states = np.array([batch.state for batch in random_batches])
            next_states = np.array([batch.next_state for batch in random_batches])
            actions = np.array([batch.action for batch in random_batches])

            # reshape
            vector_actions = (np.eye(7)[actions]).reshape((batch_size, 7, 1))
            next_states = next_states.reshape((batch_size, 6, 7, 1))
            states = states.reshape((batch_size, 6, 7, 1))

            # get next action from network
            new_q = self.get_next_Q(next_states)
            action_table = np.argsort(np.squeeze(new_q,axis=-1), axis=1)
            best_q = np.zeros((batch_size))
            predicted_action = np.zeros((batch_size))

            # replace invalid moves
            for single_batch in np.arange(batch_size):
                single_example = action_table[single_batch]
                for action in single_example:
                    if next_states[single_batch, 0, action, 0] == 0:
                        best_q[single_batch] = new_q[single_batch, action, 0]
                        predicted_action[single_batch] = action
                        break

            fixed_rewards = rewards + self.gamma * best_q

            # illegal moves
            invalid = np.zeros((batch_size,))
            invalid = 0.0 * invalid + 1.0

            # train
            feed_dict = {self.q_network.rewards: fixed_rewards, self.q_network.actions_holder: vector_actions,
                         self.q_network.boards: states, self.q_network.punishment: invalid}
            _, loss, net = self.q_network.session.run([self.q_network.optimizer, self.q_network.loss,
                                                       self.q_network.q_vals], feed_dict=feed_dict)
            if DEBUG and round % 100 == 0:
                self.log("ROUND {} loss {}".format(round, loss))

        except Exception as ex:
            print("Exception in learn: %s %s" % (type(ex), ex))

    def get_next_Q(self, curr_state):
        return self.q_network.q_vals.eval(feed_dict={self.q_network.boards: curr_state}, session=self.q_network.session)

    def get_middle_state(self, prev_state, prev_action):
        state = np.copy(prev_state)
        row = np.max(np.where(self.state[:, prev_action] == 0))
        state[row, prev_action] = -1
        return state

    def update_rates(self):
        # learning rate
        if self.q_network.lr > 0.05:
            self.q_network.lr -= self.learning_decay
        # exploration-exploitation e-greedy
        if self.epsilon > 0.01:
            self.epsilon -= self.epsilon_decay

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

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        # log game state
        if DEBUG:
            if self._round_started:
                # self.log("ROUND {} action {}\ngame state {}".format(round, prev_action, new_state))
                if reward != 0:
                    self._round_started = False

        try:
            # clear memory if full
            if len(self.transitions_memory) >= self.memory_limit:
                self.transitions_memory.popleft()

            # update learning rate
            if round % 1000 == 0:
                self.update_rates()

            new_state = self.normalize_board(new_state)

            if prev_action is not None and prev_state is not None:
                prev_state = self.normalize_board(prev_state)

                # store parameters in memory
                self.transitions_memory.append(TransitionBatch(prev_state, prev_action, reward, new_state))

            # use epsilon greedy
            if np.random.rand() < self.epsilon:

                # choose random action
                action = self.get_random_action(new_state)

            else:
                # get next action from network
                action = self.get_qNet_action(new_state)

        except Exception as ex:
            print("Exception in act: %s %s" %(type(ex), ex))
            action = np.random.choice(np.arange(NUM_ACTION))

        finally:
            return action

    def get_random_action(self, new_state):
        action = np.random.choice(np.arange(NUM_ACTION))

        # avoid illegal moves
        invalid = new_state[0, action] != 0
        while invalid:
            action = np.random.choice(np.arange(NUM_ACTION))  # TODO without illegal actions
            invalid = new_state[0, action] != 0
        return action

    def get_qNet_action(self, new_state):
        new_state = new_state.reshape(1, 6, 7, 1)
        q_values = self.get_next_Q(new_state)
        action_table = np.flipud(np.argsort(q_values, axis=1))

        for action in action_table[0, :, 0]:
            if new_state[0, 0, action, 0] == 0:
                return action

    def save_model(self):
        weights = [self.q_network.session.run(v) for v in tf.trainable_variables()]
        print("Model saved to %s" % self.save_to)
        return weights, self.save_to

    def init_run(self):
        # store all transition batches seen during game {round_num: transition_batch}
        self.transitions_memory = deque()

        # load stored model
        if self.load_from:
            with open(self.load_from, 'rb') as f:
                weights = pickle.load(f)
                for v, w in zip(tf.trainable_variables(), weights):
                    self.q_network.session.run(v.assign(w))
            print("Model loaded from %s" % self.load_from)

        self.q_network = PolicyNetwork(self.learning_rate, epochs=5, batches_per_epoch=self.batch_size)

        self.log("SmartPlayer id: {}".format(self.id))

        # for debugging
        self._round_started = False