import numpy as np
from policies.base_policy import Policy
from policies.topologies import PolicyNetwork
import random
import time
from collections import deque
import tensorflow as tf


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
        policy_args['memory_limit'] = int(policy_args['memory_limit']) if 'memory_limit' in policy_args else 5000
        policy_args['save_path'] = policy_args['save_path'] if 'save_path' in policy_args else 'model/Smart'
        policy_args['load_path'] = policy_args['load_path'] if 'load_path' in policy_args else 'model/Smart'
        return policy_args

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        try:

            # set batch size
            if self.batch_size < len(self.transitions_memory):
                batch_size = self.batch_size
            else:
                batch_size = int(self.batch_size / 2)
            if batch_size > len(self.transitions_memory):
                return

            # select random batch from memory
            random_batches = random.sample(self.transitions_memory, batch_size)

            rewards = np.array([batch.reward for batch in random_batches])
            states = np.array([batch.state for batch in random_batches])
            next_states = np.array([batch.next_state for batch in random_batches])
            actions = np.array([batch.action for batch in random_batches])

            vector_actions = (np.eye(7)[actions]).reshape((batch_size, 7, 1))
            next_states = next_states.reshape((batch_size, 6, 7, 1))
            states = states.reshape((batch_size, 6, 7, 1))

            new_q = self.get_next_Q(next_states)
            action_table = np.argsort(new_q, axis=1)
            best_q = np.max(new_q[:, 0]).reshape(batch_size)
            for sing_batch in new_q:
                action_table = action_table[sing_batch]
                for action in action_table:
                    if next_states[0, action] == 0:
                        best_q[sing_batch,:] = action
            best_q = np.max(new_q[:,action]).reshape(batch_size)
            predicted_action = np.argmax(new_q, axis=1).reshape(batch_size)
            fixed_rewards = rewards + self.gamma * best_q * np.abs(rewards)

            # illegal moves
            invalid = (next_states[np.arange(batch_size), 0, predicted_action, 0] != 0)
            invalid = 0.0 * invalid + 1.0

            # train
            feed_dict = {self.q_network.rewards: fixed_rewards, self.q_network.actions_holder: vector_actions,
                         self.q_network.boards: states, self.q_network.punishment: invalid}
            _, loss, net = self.q_network.session.run([self.q_network.optimizer, self.q_network.loss,
                                                       self.q_network.q_vals], feed_dict=feed_dict)
            # print(loss )

        except Exception as ex:
            print("Exception in learn: %s %s" % (type(ex), ex))

    def get_next_Q(self, curr_state):
        return self.q_network.q_vals.eval(feed_dict={self.q_network.boards: curr_state}, session=self.q_network.session)

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        try:
            # clear memory if full
            if len(self.transitions_memory) >= self.memory_limit:
                self.transitions_memory.popleft()

            # update learning rate
            if round % 1000 == 0:
                if self.q_network.lr > 0.05:
                    self.q_network.lr -= 0.005
                if self.epsilon > 0.01:
                    self.epsilon -= 0.001

            if reward != 0:
                self.actions = []

            # normalize board
            new_state = np.copy(new_state)
            if self.id == 1:
                new_state[np.where(new_state == 2)] = -1
            else:
                new_state[np.where(new_state == 1)] = -1
                new_state[np.where(new_state == 2)] = 1


            if prev_action is not None and prev_state is not None:
                prev_state = np.copy(prev_state)

                if self.id == 1:
                    prev_state[np.where(prev_state == 2)] = -1
                else:
                    prev_state[np.where(prev_state == 1)] = -1
                    prev_state[np.where(prev_state == 2)] = 1

                # store parameters in memory
                self.transitions_memory.append(TransitionBatch(prev_state, prev_action, reward, new_state))

            # use epsilon greedy
            if np.random.rand() < self.epsilon:
                # choose random action
                action = np.random.choice(np.arange(NUM_ACTION))

                # avoid illegal moves
                invalid = new_state[0, action] != 0
                while invalid:
                    action = np.random.choice(np.arange(NUM_ACTION))  # TODO without illegal actions
                    invalid = new_state[0, action] != 0

            else:
                # get next action from network
                new_state = new_state.reshape(1, 6, 7, 1)
                q_values = self.get_next_Q(new_state)
                action_table = np.flipud(np.argsort(q_values,axis=1))
                for action in action_table:
                    if new_state[0, action] == 0:
                        break
                # print("=Board")
                # print(new_state.reshape(6,7))
                # print("=Action=")
                # print(action)
                self.actions.append(action)

        except Exception as ex:
            print("Exception in act: %s %s" %(type(ex), ex))

            action = np.random.choice(np.arange(NUM_ACTION))
        finally:
            return action

    def save_model(self):
        return [], self.save_path
        return tf.trainable_variables(), self.save_path

    def init_run(self):
        # store all transition batches seen during game {round_num: transition_batch}
        self.transitions_memory = deque()
        self.actions = []
        self.q_network = PolicyNetwork(self.learning_rate, epochs=5, batches_per_epoch=self.batch_size)


