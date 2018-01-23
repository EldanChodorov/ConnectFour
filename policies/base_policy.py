import sys
import multiprocessing as mp
import traceback

import numpy as np
np.set_printoptions(threshold=np.nan)

POLICIES = {}

EMPTY_VAL = 0


def collect_policies():
    """
    internal function for collecting the policies in the folder.
    """
    if POLICIES: return POLICIES # only fill on first function call
    for mname in sys.modules:
        if not mname.startswith('policies.policy'): continue
        # if not mname.startswith('policies'): continue
        mod = sys.modules[mname]
        for cls_name in dir(mod):
            try:
                if cls_name != 'Policy':
                    cls = mod.__dict__[cls_name]
                    if issubclass(cls, Policy): POLICIES[cls_name] = cls
            except TypeError:
                pass
    return POLICIES


def build(policy_string):
    """
    internal function for building the desired policies when running the game.
    :param policy_string: the policy string entered when running the game.
    """
    available_policies = collect_policies()

    name, args = policy_string.split('(')
    name = name.replace(" ", "")
    if name not in available_policies:
        raise ValueError('no such policy: %s' % name)
    P = available_policies[name]
    kwargs = dict(tuple(arg.split('=')) for arg in args[:-1].split(',') if arg)
    return P, kwargs


class Policy(mp.Process):
    """
    The Policy class, representing the policy that an agent in the game has.
    The policy runs on a different process and communicates with the agent
    using queues.
    """

    def __init__(self, policy_args, stateq, actq, modelq, logq, id, mode, game_duration):
        """
        initialize the policy.
        :param policy_args: the arguments for the specific policy to be added as members.
        :param stateq: the state queue for communication with the game.
        :param actq: the action queue for communication with the game.
        :param modelq: the model queue for communication with the game.
        :param logq: the log queue for communication with the game.
        :param id: the player ID in the game (used to understand the board states)
        :param mode: 'train' if this is a training session (you can use this to decide about
                    exploration vs exploitation.
        """
        mp.Process.__init__(self)
        self.sq = stateq
        self.aq = actq
        self.mq = modelq
        self.lq = logq
        self.id = id
        self.mode = mode
        self.game_duration = game_duration
        self.__dict__.update(self.cast_string_args(policy_args))


    def log(self, msg, type=' '):
        """
        A logging function you can use for debugging and anything else you need.
        Given the message you want to log and its type, the function makes sure
        that it is added to the log file.
        :param msg: what to write in the log.
        :param type: the type of the message (e.g. "error" or "debug"...)
        """
        self.lq.put((str(self.id), type, msg))


    def get_legal_moves(self, board_state):
        """
        Given a board state, the function returns the legal moves on that board.
        :param board_state: the board state received from the game.
        :return: array of legal moves.
        """

        legal_cols = np.array(np.where(board_state[0, :] == EMPTY_VAL))
        return np.reshape(legal_cols, (legal_cols.size,))


    def run(self):
        """
        The function that the policy runs as a separate process. It checks the
        queue repeatedly for new game states and returns the appropriate actions
        according to the policy. When the accepted state is of a game which has
        ended, the policy runs the "learn" subroutine.
        The policy also listens for requests to save its model.
        The process runs indefinitely, until killed by the Game process.
        """
        try:
            self.init_run()
            for input in iter(self.sq.get, None):
                if input == 'save_model':
                    self.mq.put(self.save_model())
                else:
                    round, prev_state, prev_action, reward, new_state, is_game_over, too_slow = input
                    if is_game_over:
                        if self.mode == 'train':
                            self.learn(round, prev_state, prev_action, reward, new_state, too_slow)
                    else:
                        self.aq.put((round, self.act(round, prev_state, prev_action, reward, new_state, too_slow)))

        except Exception as e:
            tb_str = traceback.format_exc(e)
            self.log("policy %s is down: %s" % (str(self), tb_str), type='error')
            for input in iter(self.sq.get, None):
                if input[0] == 'get_state': self.aq.put(None)


    def cast_string_args(self, policy_args):
        """
        this function casts arguments passed during policy construction to their proper types/names.
        :param policy_args: a arg -> string value map as received in command line, notice that the "load_from" and
                            "save_to" are special arguments passing a file path that can be used for initialization
        :return: A map of string -> value after casting to useful objects, these will be added as members to the policy
        """
        raise NotImplementedError


    def init_run(self):
        """
        this function is called right after the initialization of the agent.
        you may use it to initialize variables that are needed for your policy,
        such as TensorFlow sessions and so on. you may also use this function
        to load your pickled model and set the variables accordingly, if the
        game uses a saved model and is not a training session.
        """
        raise NotImplementedError


    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        """
        the function for learning and improving the policy. it accepts the
        state-action-reward needed to learn from the final move of the game,
        and from that (and other state-action-rewards saved previously) it
        may improve the policy.
        :param round: the round of the game.
        :param prev_state: the previous state from which the policy acted.
        :param prev_action: the previous action the policy chose.
        :param reward: the reward given by the environment following the previous action.
        :param new_state: the new state that the agent is presented with, following the previous action.
                          This is the final state of the round.
        :param too_slow: true if the game didn't get an action in time from the
                        policy. use this to make your computation time smaller
                        by lowering the batch size for example...
        """
        raise NotImplementedError


    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        """
        the function for choosing an action, given current state.
        it accepts the state-action-reward needed to learn from the previous
        move (which it can save in a data structure for future learning), and
        accepts the new state from which it needs to decide how to act.
        :param round: the round of the game.
        :param prev_state: the previous state from which the policy acted.
        :param prev_action: the previous action the policy chose.
        :param reward: the reward given by the environment following the previous action.
        :param new_state: the new state that the agent is presented with, following the previous action.
        :param too_slow: true if the game didn't get an action in time from the
                policy. use this to make your computation time smaller
                by lowering the batch size for example...
        :return: an action (from Policy.Actions) in response to the new_state.
        """
        raise NotImplementedError


    def save_model(self):
        """
        A function you must implement, which returns your model, along with
        where to save your model to. The model object need to be picklable (like
        a list, for example).
        :return: a tuple: (model, save_to)
                The "model" is the policy model (e.g. for future initialization)
                The "save_to" is the path where your model will be saved. This should
                be the save_to parameter you are given from the command line
                when the game starts.
        """
        raise NotImplementedError
