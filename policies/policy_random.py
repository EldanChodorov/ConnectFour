from policies import base_policy as bp
import numpy as np
import pickle

EMPTY_VAL = 0

class RandomAgent(bp.Policy):
    """
    A random agent, which plays random legal actions.
    """

    def cast_string_args(self, policy_args):

        policy_args['example'] = int(policy_args['example']) if 'example' in policy_args else 0
        policy_args['save_to'] = str(policy_args['save_to']) if 'save_to' in policy_args else 'random.model.pkl'
        return policy_args


    def init_run(self):

        try:
            state = pickle.load(open(self.load_from, 'rb'))
            self.reward_sum = state[0]
            self.history = state[1]
        except:
            self.reward_sum = 0
            self.history = []


    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        self.reward_sum += reward
        self.history = []


    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        try:
            self.reward_sum += reward

            if prev_state is not None:
                self.history.append((prev_state, prev_action, reward))

            legal_cols = np.array(np.where(new_state[0, :] == EMPTY_VAL))
            legal_cols = np.reshape(legal_cols, (legal_cols.size,))
            return np.random.choice(legal_cols)

        except:
            self.log("Oh No! Exception!", 'EXCEPTION')


    def save_model(self):

        return self.reward_sum, self.save_to
