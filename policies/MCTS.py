import numpy as np

NUM_ACTIONS = 7
PLAYER_ID = 1
ENEMY_ID = -1


class Node:
    def __init__(self, state, policy, value, constant):
        self.state = state

        self.num_visits = np.zeros(NUM_ACTIONS, np.int32)
        self.weight = np.zeros(NUM_ACTIONS, np.float32)
        self.Q_val = np.zeros(NUM_ACTIONS, np.float32)
        self.child = [None] * NUM_ACTIONS

        self.policy = policy
        self.value = value
        self.constant = constant

    def pick_child(self):
        U = self.constant * self.policy * np.sqrt(self.num_visits.sum()) / (1. + self.num_visits)
        return (self.Q_val + U).argmax()

    def propogate_reward(self, a, v):
        self.num_visits[a] += 1
        self.weight[a] += v
        self.Q_val[a] = self.weight[a] / self.num_visits[a]

    def probabilities(self, tau=1.):
        if tau == 0.:
            p = np.zeros(NUM_ACTIONS, np.float32)
            p[self.num_visits.argmax()] = 1.
            return p
        else:
            p = self.num_visits ** (1. / tau)
            return p / p.sum()


class Tree:
    def __init__(self, simulation, policy, max_depth, constant):
        self.policy = policy
        self.sim = simulation
        self.prior = None
        self.max_depth = max_depth
        self.constant = constant
        self.root = self.create_node(None)

    def create_node(self, parent):
        state = self.sim.get_state()
        if parent:
            P, V = parent.policy, parent.value
        else:
            P, V = 0.5, 0.5
        # P, V = self.prior(state)  # TODO what is prior??
        return Node(state, P, V, self.constant)

    def create_child(self, node, a):
        action = self.policy.get_action(node.state)
        self.sim.set_state(node.state)
        self.sim.step(action, a)

        if self.sim.win == PLAYER_ID:
            node.child[a] = PLAYER_ID
        elif self.sim.win == ENEMY_ID:
            node.child[a] = ENEMY_ID
        else:
            node.child[a] = self.create_node(node)

    def select(self):
        stack = []
        node = self.root

        for i in range(self.max_depth):
            a = node.pick_child()
            stack.append((node, a))

            if node.child[a] is None:
                self.create_child(node, a)

            if node.child[a] in [1, -1]:
                v = node.child[a]
                break

            node = node.child[a]
            v = node.value

        for node, a in stack:
            node.num_visits[a] += 1
            node.weight[a] += v
            node.Q_val[a] = node.weight[a] / node.num_visits[a]

    def search(self, num):
        for i in range(num):
            self.select()

    def step(self, a):
        if self.root.child[a] is None:
            self.create_child(self.root, a)

        self.root = self.root.child[a]

    def done(self):
        return self.root in [ENEMY_ID, PLAYER_ID]


class UniformPolicy(object):

    @staticmethod
    def get_action(state):
        # state: numpy array
        action = np.random.choice(np.arange(NUM_ACTIONS))

        # avoid illegal moves
        invalid = state[0, action] != 0
        while invalid:
            action = np.random.choice(np.arange(NUM_ACTIONS))
            invalid = state[0, action] != 0
        return action


class Simulator:

    def __init__(self, initial_state):
        self.state = initial_state
        self.id = PLAYER_ID
        self.enemy_id = ENEMY_ID
        self.win = None

    def get_state(self):
        return self.state

    def insert(self, column):
        # make sure column isn't full
        invalid = self.state[0, column] != 0
        while invalid:
            column = np.random.choice(np.arange(NUM_ACTIONS))
            invalid = self.state[0, column] != 0
        row = np.max(np.where(self.state[:, column] == 0))
        self.state[row, column] = self.id
        return row

    def step(self, first_action, second_action):
        row_inserted = self.insert(first_action)
        if self.check_win(self.id, first_action, row_inserted):
            self.win = self.id
        row_inserted = self.insert(second_action)
        if self.check_win(self.enemy_id, second_action, row_inserted):
            self.win = self.enemy_id

    def set_state(self, state):
        self.state = state

    def check_win(self, player_id, curr_col, curr_row):

        rows, cols = self.state.shape
        win_mask = np.ones(4)

        # check horizontal:
        vec = self.state[curr_row, :] == player_id
        if np.any(np.convolve(win_mask, vec, mode="valid") == 4):
            return True

        # check vertical:
        vec = self.state[:, curr_col] == player_id
        if np.any(np.convolve(win_mask, vec, mode="valid") == 4):
            return True

        # check diagonals:
        vec = np.diagonal(self.state, curr_col-curr_row) == player_id
        if np.any(np.convolve(win_mask, vec, mode="valid") == 4):
            return True
        vec = np.diagonal(np.fliplr(self.state), cols-curr_col-1-curr_row) == player_id
        if np.any(np.convolve(win_mask, vec, mode="valid") == 4):
            return True

        return False
