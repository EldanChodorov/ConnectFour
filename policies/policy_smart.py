import numpy as np
from policies.base_policy import Policy
from policies.topologies import PolicyNetwork
import random
import time

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
        return {}

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        try:
            if self.batch_size < len(self.transitions_memory):

                # select random batch from memory
                random_batches = random.sample(list(self.transitions_memory.values()), self.batch_size)

                rewards = np.array([batch.reward for batch in random_batches])
                states = np.array([batch.state for batch in random_batches])
                next_states = np.array([batch.next_state for batch in random_batches])
                actions = np.array([batch.action for batch in random_batches])

                vector_actions = (np.eye(7)[actions]).reshape((self.batch_size, 7, 1))
                next_states = next_states.reshape((self.batch_size, 6, 7, 1))
                states = states.reshape((self.batch_size, 6, 7, 1))

                new_q = self.get_next_Q(next_states)
                predicted_action = np.argmax(new_q, axis=1).reshape(self.batch_size)
                fixed_rewards = rewards + self.gamma * predicted_action

                # illegal moves
                invalid = (next_states[np.arange(self.batch_size), 0, predicted_action, 0] != 0)
                invalid = 0.1 * invalid + 1.0

                # train
                feed_dict = {self.q_network.rewards: fixed_rewards, self.q_network.actions_holder: vector_actions,
                             self.q_network.boards: states, self.q_network.punishment:invalid}
                _, loss, net = self.q_network.session.run([self.q_network.optimizer, self.q_network.loss,
                                                           self.q_network.x], feed_dict=feed_dict)

                if round % 5 == 0:
                    import tensorflow as tf
                    gr = tf.get_default_graph()
                    print(gr.get_tensor_by_name("conv1/kernel:0").eval(session=self.q_network.session))
                    print(gr.get_tensor_by_name("conv2/kernel:0").eval(session=self.q_network.session))
                    print(gr.get_tensor_by_name("conv2/kernel:0").eval(session=self.q_network.session))

        except Exception as ex:
            print("Exception in learn: %s %s" % (type(ex), ex))

    def get_next_Q(self, curr_state):
        # return self.q_network.session.run(self.q_network.x, feed_dict={self.q_network.x: curr_state})
        return self.q_network.x.eval(feed_dict={self.q_network.boards: curr_state}, session=self.q_network.session)

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        try:
            action = None
            if reward != 0:
                print(self.actions)
                self.actions = []

            # store parameters in memory
            if prev_action is not None and prev_state is not None:
                self.transitions_memory[round] = TransitionBatch(prev_state, prev_action, reward, new_state)

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
                action = np.argmax(q_values)
                self.actions.append(action)

        except Exception as ex:
            print("Exception in act: %s %s" %(type(ex), ex))
            action = np.random.choice(np.arange(NUM_ACTION))
        finally:
            return action

    def save_model(self):
        return [], 'models'

    def init_run(self):
        # store all transition batches seen during game {round_num: transition_batch}
        self.transitions_memory = dict()
        self.actions = []

        self.batch_size = 5
        self.q_network = PolicyNetwork(0.1, 5, self.batch_size)

        self.epsilon = 0.1
        self.gamma = 0.95

