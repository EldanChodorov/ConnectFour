import numpy as np
from policies.base_policy import Policy
from policies.topologies import PolicyNetwork
import random


class TransitionBatch:
    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state


NUM_ACTION = 7


class SmartPolicy(Policy):

    def __init__(self, policy_args, stateq, actq, modelq, logq, id, mode, game_duration):
        Policy.__init__(self, policy_args, stateq, actq, modelq, logq, id, mode, game_duration)

        # store all transition batches seen during game {round_num: transition_batch}
        self.transitions_memory = dict()

        self.batch_size = 50
        self.q_network = PolicyNetwork(0.1, 5, self.batch_size)

        self.epsilon = 0.2
        self.gamma = 0.3

    def cast_string_args(self, policy_args):
        return {}

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        # select random batch from memory
        random_batches = random.sample(self.transitions_memory.values(), self.batch_size)

        rewards = np.array([batch.reward for batch in random_batches])
        states = [batch.state for batch in random_batches]
        next_states = [batch.next_state for batch in random_batches]
        actions = [batch.action for batch in random_batches]

        predicted_state = self.get_next_state(next_states)
        fixed_rewards = rewards + self.gamma * predicted_state

        # train
        _, loss, net = self.q_network.session.run([self.q_network.optimizer, self.q_network.loss,
                                                   self.q_network.x], feed_dict={self.rewards: fixed_rewards,
                                                    self.actions_holder: actions, self.boards: states})

    def get_next_state(self, curr_state):
        return self.q_network.session.run(feed_dict={self.q_network.x: curr_state})

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        # store parameters in memory
        self.transitions_memory[round] = TransitionBatch(prev_state, prev_action, reward, new_state)

        # use epsilon greedy
        if np.random.rand() < self.epsilon:
            # choose random action
            action = np.random.choice(np.arange(NUM_ACTION))
        else:
            # get next action from network
            q_values = self.get_next_state(new_state)
            action = np.argmax(q_values)

        return action

    def save_model(self):
        return [1]

    def init_run(self):
        return None
