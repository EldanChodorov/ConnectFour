import sys

INITIALIZERS = {}


def collect_initializers():
    if INITIALIZERS: return INITIALIZERS # only fill on first function call
    for mname in sys.modules:
        if not mname.startswith('initializers.initializer'): continue
        mod = sys.modules[mname]
        for cls_name in dir(mod):
            try:
                if cls_name != 'Initializer':
                    cls = mod.__dict__[cls_name]
                    if issubclass(cls, Initializer): INITIALIZERS[cls_name] = cls
            except TypeError:
                pass
    return INITIALIZERS


def find_initializer(initializer_string):
    available_initializers = collect_initializers()
    if initializer_string not in available_initializers:
        raise ValueError('no such initializer: %s' % initializer_string)
    return available_initializers[initializer_string]


class Initializer(object):

    def initialize(self, game_num, starting_player):
        """
        Given a game iteration number, return the board to start the game from.
        :param game_num: game iteration number
        :return: board - np.matrix of size (6,7)
        """
        raise NotImplementedError