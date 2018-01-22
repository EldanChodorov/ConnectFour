import numpy as np
from policies.base_policy import Policy
from policies.topologies import PolicyNetwork
import random
import time
from collections import deque
import tensorflow as tf
import pickle
from policies.MCTS import Node, Tree, UniformPolicy, Simulator

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


class MCTSAgent(Policy):

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

    def last_round_batches(self):
        # select random batches from memory
        random_batches = random.sample(self.transitions_memory, self.batch_size)

        # extract features from batches
        rewards = np.array([batch.reward for batch in random_batches])
        states = np.array([batch.state for batch in random_batches])
        next_states = np.array([batch.next_state for batch in random_batches])
        actions = np.array([batch.action for batch in random_batches])

        return rewards, states, next_states, actions

    def learnMCTS(self):

        rewards_batch, states_batch, next_states_batch, actions_batch = self.last_round_batches()
        batch_size = rewards_batch.shape[0]

        all_states, all_probs, all_rewards = [], [], []

        for i in range(batch_size):

            states = np.zeros((self.learn_tree_depth, 6, 7), np.float32)
            probs = np.zeros((self.learn_tree_depth, NUM_ACTION), np.float32)

            state = states_batch[i]
            self.mcts.sim.set_state(state)

            for d in range(self.learn_tree_depth):
                self.mcts.search(num=3)
                p = self.mcts.root.probabilities()

                states[d] = self.mcts.root.state
                probs[d] = p
                d += 1

                a = np.random.choice(NUM_ACTION, 1, p=p)[0]
                self.mcts.step(a)

                if self.mcts.done():
                    break

            rewards = np.zeros((1, d), np.float32)
            if self.mcts.done():
                rewards[:] = self.mcts.root

            all_states.append(states)
            all_probs.append(probs)
            all_rewards.append(rewards)

        return all_states, all_probs, all_rewards

    def learn_iteration(self, result_states, result_probs, result_rewards):
        """Perform a single Alpha-Pong-Zero iteration."""
        try:
            states = (np.concatenate(result_states))[:, :, :, None]
            probs = (np.concatenate(result_probs))[:, :, :, None]
            rewards = (np.concatenate(result_rewards))[:, :, :, None]

            feed_dict = {self.q_network.boards: states, self.q_network.probabilities: probs, self.q_network.rewards: rewards}
            self.q_network.session.run(self.optimizer, feed_dict=feed_dict)
        except Exception as ex:
            print(ex)

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        try:

            # TODO store attrs in memory

            results = self.learnMCTS()
            self.learn_iteration(*results)

        except Exception as ex:
            print("Exception in learn: %s %s" % (type(ex), ex))

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

    def update_rates(self):
        # learning rate
        if self.q_network.lr > 0.05:
            self.q_network.lr -= self.learning_decay
        # exploration-exploitation e-greedy
        if self.epsilon > 0.01:
            self.epsilon -= self.epsilon_decay

    def get_random_action(self, new_state):
        action = np.random.choice(np.arange(NUM_ACTION))

        # avoid illegal moves
        invalid = new_state[0, action] != 0
        while invalid:
            action = np.random.choice(np.arange(NUM_ACTION))  # TODO without illegal actions
            invalid = new_state[0, action] != 0
        return action

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

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
            return action

        except Exception as ex:
            print("Exception in act: %s %s" %(type(ex), ex))
            return np.random.choice(np.arange(NUM_ACTION))

    def get_next_Q(self, curr_state):
        return self.q_network.q_vals.eval(feed_dict={self.q_network.boards: curr_state}, session=self.q_network.session)

    def get_qNet_action(self, new_state):
        new_state = new_state.reshape(1, 6, 7, 1)
        q_values = self.get_next_Q(new_state)
        action_table = np.flipud(np.argsort(q_values, axis=1))

        for action in action_table[0, :, 0]:
            if new_state[0, 0, action, 0] == 0:
                return action

    def save_model(self):
        print("Model saved to %s" % self.save_to)
        return [], self.save_to

    def init_run(self):
        self.log("MCTS Agent id: {}".format(self.id))

        # store all transition batches seen during game {round_num: transition_batch}
        self.transitions_memory = deque()

        # load stored model
        if self.load_from:
            with open(self.load_from, 'rb') as f:
                pass

        self.q_network = PolicyNetwork(self.learning_rate, epochs=5, batches_per_epoch=self.batch_size)

        self.learn_depth = 1
        self.learn_tree_depth = 2

        # network parameters
        self.evaluation = tf.tanh(self.q_network.q_vals)
        self.cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                            logits=tf.squeeze(self.q_network.q_vals, [-1]), labels=self.q_network.probabilities))
        self.evaluation_loss = tf.reduce_mean((self.q_network.rewards - self.evaluation) ** 2)
        self.total_loss = self.evaluation_loss + self.cross_entropy_loss
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

        self.tree_depth = 3
        simulator = Simulator(np.zeros((6, 7)))
        self.mcts = Tree(simulator, UniformPolicy(), max_depth=self.tree_depth, constant=0.3)