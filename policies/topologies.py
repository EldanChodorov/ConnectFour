'''
Test different neural network topologies.
'''

import tensorflow as tf
import numpy as np


class PolicyNetwork:

    def __init__(self, learn_rate, epochs, batches_per_epoch):
        self.lr = learn_rate
        self.epochs = epochs
        self.batches_per_epoch = batches_per_epoch

        self.session = None

        # Q network variables
        self.actions_holder = tf.placeholder(tf.float32, shape=[None, 7, 1], name="action_holder")
        self.boards = tf.placeholder(tf.float32, shape=[None, 6, 7, 2], name="boards")
        self.winning_vec = tf.placeholder(tf.float32, shape=[None, 7, 2], name="actions_vector")
        self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")

        self.q_vals = self.network_run(self.boards, self.winning_vec)
        # self.c_vals = self.C_network_run(self.boards)

        self.action = tf.reduce_max(self.q_vals * self.actions_holder, reduction_indices=1)
        self.loss = tf.reduce_mean(tf.pow(self.rewards - self.action, 2))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        # self.probabilities = tf.nn.softmax(tf.squeeze(self.q_vals, [-1]))

        self.init = tf.initialize_all_variables()
        self.session = tf.Session()
        with self.session.as_default():
            self.session.run(self.init)

    def set_C_weights(self):

        # USE ONLY WITH NET 11
        all_vars = tf.trainable_variables()
        def get_var(name):
            for i in range(len(all_vars)):
                if all_vars[i].name.startswith(name):
                    return all_vars[i]
            return None

        matching_vars = []  # tuples (q_var, c_var)
        for i in range(5):
            weights_var = 'net11/fc{}/weights'.format(i)
            C_weights_var = 'C_net11/fc{}/weights'.format(i)
            matching_vars.append((get_var(weights_var), get_var(C_weights_var)))
            biases_var = 'net11/fc{}/biases'.format(i)
            C_biases_var = 'net11/fc{}/biases'.format(i)
            matching_vars.append((get_var(biases_var), get_var(C_biases_var)))

        for var, Cvar in matching_vars:
            # qWeight = self.session.run(var)
            if Cvar:
                try:
                    self.session.run(Cvar.assign(var))
                except Exception as ex:
                    print(ex)

    def weight(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def bias(self, shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    def C_network_run(self, feed_inputs):
        # USE ONLY C NETWORKS!
        return self.C_net_try11(feed_inputs)

    def network_run(self, boards, action_vec):
        '''

        :param boards: (?, 6, 7, 2, 1)
        :param action_vec: (?, 7)
        :return:
        '''
        return self.net_try17(boards, action_vec)

    def network_run2(self, feed_inputs):
        '''
        Runs the given input through the network architecture.
        Architecture:
        ...
        :param feed_inputs: matrix [numpy.ndarray] 6x7
        :return: result from network. - [1X7]
        '''
        return self.net_try11(feed_inputs)

    def net_try1(self, inputs):
        print("NETWORK 1")
        conv_layer1 = tf.nn.relu(tf.nn.conv2d(inputs, 8, [5, 5], padding='same', name="conv1"))
        conv_layer2 = tf.nn.relu(tf.nn.conv2d(conv_layer1, 16, [3, 3], padding='same', name="conv2"))
        conv_layer3 = tf.nn.relu(tf.nn.conv2d(conv_layer2, 32, [3, 3], padding='same', name="conv3"))
        sum_layer1 = tf.reduce_sum(conv_layer3, reduction_indices=1)
        fully_connected1 = tf.contrib.layers.fully_connected(sum_layer1, 1, activation_fn=None)
        return fully_connected1




    def net_try3(self, inputs):
        print("NETWORK 3")

        conv_layer1 = tf.layers.conv2d(inputs, 8, [5, 5], padding='same', activation=tf.nn.relu)
        conv_layer2 = tf.layers.conv2d(conv_layer1, 16, [5, 5], padding='same', activation=tf.nn.relu)
        conv_layer3 = tf.layers.conv2d(conv_layer2, 32, [3, 3], padding='same', activation=tf.nn.relu)
        conv_layer4 = tf.layers.conv2d(conv_layer3, 64, [3, 3], padding='same', activation=tf.nn.relu)
        sum_layer1 = tf.reduce_sum(conv_layer4, reduction_indices=1)
        fully_connected1 = tf.layers.Dense(sum_layer1, 1)
        return fully_connected1

    def net_try4(self, inputs):
        print("NETWORK 4")

        input_shape = inputs.get_shape().as_list()

        weight1 = self.weight([input_shape[1] * input_shape[2] * input_shape[3], 10])
        bias1 = self.bias([10])

        input_flat = tf.reshape(inputs, [-1, input_shape[1] * input_shape[2] * input_shape[3]])

        fully_connected1 = tf.nn.relu(tf.matmul(input_flat, weight1) + bias1)

        weight2 = self.weight([10, 7])
        bias2 = self.bias([7])

        fully_connected2 = tf.nn.relu(tf.matmul(fully_connected1, weight2) + bias2)

        final = tf.expand_dims(fully_connected2, -1)
        return final


        # fully_connected3 = tf.layers.dense(conv_layer3, 16)
        # conv_layer4 = tf.layers.conv2d(fully_connected3, 16, [3, 3], padding='same',
        #                                        activation=tf.nn.relu,
        #                                        name="conv4")
        # conv_layer5 = tf.layers.conv2d_transpose(conv_layer4,4,[3,3],2)
        # sum_layer1 = tf.reduce_sum(conv_layer5, reduction_indices=1)
        # fully_connected4 = tf.layers.dense(sum_layer1, 1, activation=None)
        # return fully_connected4

    def net_try5(self, inputs):
        print("NETWORK 5")

        input_shape = inputs.get_shape().as_list()
        weight1 = self.weight([input_shape[1] * input_shape[2] * input_shape[3], 10])
        bias1 = self.bias([10])

        input_flat = tf.reshape(inputs, [-1, input_shape[1] * input_shape[2] * input_shape[3]])

        fully_connected1 = tf.nn.relu(tf.matmul(input_flat, weight1) + bias1)

        weight2 = self.weight([10, 7])
        bias2 = self.bias([7])

        fully_connected2 = tf.matmul(fully_connected1, weight2) + bias2

        final = tf.expand_dims(fully_connected2, -1)

        return final

    def net_try6(self, inputs):
        print("NETWORK 6")

        input_shape = inputs.get_shape().as_list()
        input_flat = tf.reshape(inputs, [-1, input_shape[1] * input_shape[2] * input_shape[3]])

        fc1 = tf.contrib.layers.fully_connected(input_flat, 128, biases_initializer=tf.random_normal_initializer(),
                                                weights_initializer=tf.random_normal_initializer())
        fc2 = tf.contrib.layers.fully_connected(fc1, 64, biases_initializer=tf.random_normal_initializer(),
                                                weights_initializer=tf.random_normal_initializer())
        fc3 = tf.contrib.layers.fully_connected(fc2, 7, biases_initializer=tf.random_normal_initializer(),
                                                weights_initializer=tf.random_normal_initializer())

        final = tf.reshape(fc3, [-1, 7, 1])
        return final

    def net_try7(self, inputs):
        print("NETWORK 7")

        input_shape = inputs.get_shape().as_list()
        input_flat = tf.reshape(inputs, [-1, input_shape[1] * input_shape[2] * input_shape[3]])

        # fc1 = tf.contrib.layers.fully_connected(input_flat, 256)
        fc2 = tf.contrib.layers.fully_connected(input_flat, 128)
        fc3 = tf.contrib.layers.fully_connected(fc2, 64)
        fc4 = tf.contrib.layers.fully_connected(fc3, 7)
        final = tf.reshape(fc4, [-1, 7, 1])
        return final

    def net_try8(self, inputs):

        input_shape = inputs.get_shape().as_list()
        input_flat = tf.reshape(inputs, [-1, input_shape[1] * input_shape[2] * input_shape[3]])

        fc1 = tf.contrib.layers.fully_connected(input_flat, 256, biases_initializer=tf.random_normal_initializer(),
                                                weights_initializer=tf.random_normal_initializer())
        fc2 = tf.contrib.layers.fully_connected(fc1, 128, biases_initializer=tf.random_normal_initializer(),
                                                weights_initializer=tf.random_normal_initializer())
        fc3 = tf.contrib.layers.fully_connected(fc2, 64, biases_initializer=tf.random_normal_initializer(),
                                                weights_initializer=tf.random_normal_initializer())
        fc4 = tf.contrib.layers.fully_connected(fc3, 32, biases_initializer=tf.random_normal_initializer(),
                                                weights_initializer=tf.random_normal_initializer())
        fc5 = tf.contrib.layers.fully_connected(fc4, 16, biases_initializer=tf.random_normal_initializer(),
                                                weights_initializer=tf.random_normal_initializer())
        fc6 = tf.contrib.layers.fully_connected(fc5, 7, biases_initializer=tf.random_normal_initializer(),
                                                weights_initializer=tf.random_normal_initializer())

        final = tf.reshape(fc6, [-1, 7, 1])
        return final

    def net_try9(self, inputs):
        print("NETWORK 9")
        input_shape = inputs.get_shape().as_list()

        input_flat = tf.reshape(inputs, [-1, input_shape[1] * input_shape[2] * input_shape[3]])

        fc1 = tf.contrib.layers.fully_connected(input_flat, 128,
                                                biases_initializer=tf.random_normal_initializer(),
                                                weights_initializer=tf.random_normal_initializer())
        fc2 = tf.contrib.layers.fully_connected(fc1, 64, biases_initializer=tf.random_normal_initializer(),
                                                weights_initializer=tf.random_normal_initializer())
        fc3 = tf.contrib.layers.fully_connected(fc2, 7, biases_initializer=tf.random_normal_initializer(),
                                                weights_initializer=tf.random_normal_initializer())

        final = tf.reshape(fc3, [-1, 7, 1])
        return final



    def net_try10(self, inputs):
        print("NETWORK 10")

        first_stat = inputs[:,0,:,:]
        first_stat_shape = first_stat.get_shape().as_list()
        outer_lyears = inputs[:,1:,:,:]
        outer_lyears= tf.reduce_max(outer_lyears,reduction_indices=2)
        outer_lyears = tf.reduce_max(outer_lyears,reduction_indices=1)
        outer_lyears_shape = outer_lyears.get_shape().as_list()
        flatten_outer = tf.reshape(outer_lyears, [-1, outer_lyears_shape[1] * outer_lyears_shape[2]])
        flatten_inner = tf.reshape(first_stat, [-1, first_stat_shape[1] * first_stat_shape[2]])
        total_flatten = tf.concat(0,[flatten_inner,flatten_outer])
        fc1 = tf.contrib.layers.fully_connected(total_flatten, 64,
                                                biases_initializer=tf.random_normal_initializer(),
                                                weights_initializer=tf.random_normal_initializer())
        fc2 = tf.contrib.layers.fully_connected(fc1, 32, biases_initializer=tf.random_normal_initializer(),
                                                weights_initializer=tf.random_normal_initializer())
        fc3 = tf.contrib.layers.fully_connected(fc2, 7, biases_initializer=tf.random_normal_initializer(),
                                                weights_initializer=tf.random_normal_initializer())

        final = tf.reshape(fc3, [-1, 7, 1])
        return final



    def net_try11(self, inputs):
        print("NETWORK 11")

        with tf.variable_scope('net11'):

            first_stat = inputs[:,0,:,:]
            first_stat_shape = first_stat.get_shape().as_list()
            outer_lyears = inputs[:,1:,:,:]
            outer_lyears= tf.reduce_max(outer_lyears, reduction_indices=2)
            outer_lyears = tf.reduce_max(outer_lyears, reduction_indices=1)
            flatten_inner = tf.reshape(first_stat, [-1, first_stat_shape[1] * first_stat_shape[2]])
            fc1 = tf.contrib.layers.fully_connected(flatten_inner, 128,biases_initializer=tf.random_normal_initializer(),
                                                    weights_initializer=tf.random_normal_initializer(), scope='fc1')
            fc2 = tf.contrib.layers.fully_connected(fc1, 64, biases_initializer=tf.random_normal_initializer(),
                                                    weights_initializer=tf.random_normal_initializer(), scope='fc2')
            fc3 = tf.contrib.layers.fully_connected(fc2, 7, biases_initializer=tf.random_normal_initializer(),
                                                    weights_initializer=tf.random_normal_initializer(), scope='fc3')

            fc_v = tf.contrib.layers.fully_connected(outer_lyears, 7,biases_initializer=tf.random_normal_initializer(),
                                                    weights_initializer=tf.random_normal_initializer(), scope='fc4',
                                   activation_fn=tf.nn.sigmoid)
            connect = fc3 * fc_v
            connect_fc = tf.contrib.layers.fully_connected(connect, 7,
                                                     biases_initializer=tf.random_normal_initializer(), scope='fc5',
                                              weights_initializer=tf.random_normal_initializer(),activation_fn=None)
            final = tf.reshape(connect_fc, [-1, 7, 1])
            return final


    def C_net_try11(self, inputs):
        print("NETWORK 11")  # DO NOT CHANGE VAR NAMES

        with tf.variable_scope('C_net11'):

            first_stat = inputs[:, 0, :, :]
            first_stat_shape = first_stat.get_shape().as_list()
            outer_lyears = inputs[:, 1:, :, :]
            outer_lyears = tf.reduce_max(outer_lyears, reduction_indices=2)
            outer_lyears = tf.reduce_max(outer_lyears, reduction_indices=1)
            flatten_inner = tf.reshape(first_stat, [-1, first_stat_shape[1] * first_stat_shape[2]])
            fc1 = tf.contrib.layers.fully_connected(flatten_inner, 32, biases_initializer=tf.random_normal_initializer(),
                                                    weights_initializer=tf.random_normal_initializer(), scope='fc1')
            fc2 = tf.contrib.layers.fully_connected(fc1, 16, biases_initializer=tf.random_normal_initializer(),
                                                    weights_initializer=tf.random_normal_initializer(), scope='fc2')
            fc3 = tf.contrib.layers.fully_connected(fc2, 7, biases_initializer=tf.random_normal_initializer(),
                                                    weights_initializer=tf.random_normal_initializer(), scope='fc3')

            fc4 = tf.contrib.layers.fully_connected(outer_lyears, 7, biases_initializer=tf.random_normal_initializer(),
                                                     weights_initializer=tf.random_normal_initializer(), scope='fc4',
                                                     activation_fn=tf.nn.sigmoid)
            connect = fc3 * fc4
            fc5 = tf.contrib.layers.fully_connected(connect, 7, scope='fc5',
                                                           biases_initializer=tf.random_normal_initializer(),
                                                           weights_initializer=tf.random_normal_initializer(), activation_fn=None)
            final = tf.reshape(fc5, [-1, 7, 1])
            return final


    def net_try12(self, boards, action_vec):
        print("NETWORK 12")

        with tf.variable_scope('net12'):

            board_shape = boards.get_shape().as_list()
            flatten_board = tf.reshape(boards, [-1, board_shape[1] * board_shape[2] * board_shape[3]])
            action_vec = tf.squeeze(action_vec, [-1])
            fc1 = tf.contrib.layers.fully_connected(flatten_board, 256, biases_initializer=tf.random_normal_initializer(),
                                                    weights_initializer=tf.random_normal_initializer(), scope='fc1',
                                                    activation_fn=tf.nn.tanh)
            fc2 = tf.contrib.layers.fully_connected(fc1, 128, biases_initializer=tf.random_normal_initializer(),
                                                    weights_initializer=tf.random_normal_initializer(), scope='fc2',
                                                    activation_fn=tf.nn.tanh)
            fc3 = tf.contrib.layers.fully_connected(fc2, 7, biases_initializer=tf.random_normal_initializer(),
                                                    weights_initializer=tf.random_normal_initializer(), scope='fc3',
                                                    activation_fn=tf.nn.tanh)

            fc_v = tf.contrib.layers.fully_connected(action_vec, 64, biases_initializer=tf.random_normal_initializer(),
                                                     weights_initializer=tf.random_normal_initializer(), scope='fc4',
                                                     activation_fn=tf.nn.tanh)
            fc_v2 = tf.contrib.layers.fully_connected(fc_v, 7, biases_initializer=tf.random_normal_initializer(),
                                                     weights_initializer=tf.random_normal_initializer(), scope='fc5',
                                                     activation_fn=tf.nn.tanh)
            connect = fc3 * fc_v2
            connect_fc = tf.contrib.layers.fully_connected(connect, 7,
                                                           biases_initializer=tf.random_normal_initializer(), scope='fc6',
                                                           weights_initializer=tf.random_normal_initializer(), activation_fn=None)
            final = tf.reshape(connect_fc, [-1, 7, 1])
            return final

    def net_try13(self, boards, winning_vec):
        vac_shape = winning_vec.get_shape().as_list()
        action_vec = tf.reshape(winning_vec, [-1, vac_shape[1] * vac_shape[2]])
        fc3 = tf.contrib.layers.fully_connected(action_vec, 7, biases_initializer=tf.random_normal_initializer(),
                                                weights_initializer=tf.random_normal_initializer(), scope='fc3',
                                                activation_fn=tf.nn.tanh)
        final = tf.reshape(fc3, [-1, 7, 1])
        return final

    def net_try14(self, boards, action_vec):
        print("NETWORK 14")

        with tf.variable_scope('net14'):
            board_shape = boards.get_shape().as_list()
            flatten_board = tf.reshape(boards, [-1, board_shape[1] * board_shape[2] * board_shape[3]])
            action_vec = tf.squeeze(action_vec, [-1])
            fc1 = tf.contrib.layers.fully_connected(flatten_board, 256, biases_initializer=tf.random_normal_initializer(),
                                                    weights_initializer=tf.random_normal_initializer(), scope='fc1',
                                                    activation_fn=tf.nn.sigmoid)
            fc2 = tf.contrib.layers.fully_connected(fc1, 128, biases_initializer=tf.random_normal_initializer(),
                                                    weights_initializer=tf.random_normal_initializer(), scope='fc2',
                                                    activation_fn=tf.nn.sigmoid)
            fc3 = tf.contrib.layers.fully_connected(fc2, 7, biases_initializer=tf.random_normal_initializer(),
                                                    weights_initializer=tf.random_normal_initializer(), scope='fc3',
                                                    activation_fn=tf.nn.sigmoid)

            # fc_v = tf.contrib.layers.fully_connected(action_vec, 64, biases_initializer=tf.random_normal_initializer(),
            #                                          weights_initializer=tf.random_normal_initializer(), scope='fc4',
            #                                          activation_fn=tf.nn.tanh)
            # fc_v2 = tf.contrib.layers.fully_connected(fc_v, 7, biases_initializer=tf.random_normal_initializer(),
            #                                           weights_initializer=tf.random_normal_initializer(), scope='fc5',
            #                                           activation_fn=tf.nn.tanh)
            final_vec = tf.concat(1,[fc3, action_vec])
            connect_fc = tf.contrib.layers.fully_connected(final_vec, 7,
                                                           biases_initializer=tf.random_normal_initializer(), scope='fc6',
                                                           weights_initializer=tf.random_normal_initializer(),
                                                           activation_fn=tf.nn.sigmoid)
            final = tf.reshape(connect_fc, [-1, 7, 1])
            return final

    def net_try15(self, boards, action_vec):
        print("NETWORK 15")

        with tf.variable_scope('net14'):
            board_shape = boards.get_shape().as_list()
            flatten_board = tf.reshape(boards, [-1, board_shape[1] * board_shape[2] * board_shape[3]])
            action_vec = tf.squeeze(action_vec, [-1])
            fc1 = tf.contrib.layers.fully_connected(flatten_board, 256, biases_initializer=tf.random_normal_initializer(),
                                                    weights_initializer=tf.random_normal_initializer(), scope='fc1',
                                                    activation_fn=tf.nn.tanh)
            fc2 = tf.contrib.layers.fully_connected(fc1, 128, biases_initializer=tf.random_normal_initializer(),
                                                    weights_initializer=tf.random_normal_initializer(), scope='fc2',
                                                    activation_fn=tf.nn.tanh)
            fc3 = tf.contrib.layers.fully_connected(fc2, 7, biases_initializer=tf.random_normal_initializer(),
                                                    weights_initializer=tf.random_normal_initializer(), scope='fc3',
                                                    activation_fn=tf.nn.tanh)

            # fc_v = tf.contrib.layers.fully_connected(action_vec, 64, biases_initializer=tf.random_normal_initializer(),
            #                                          weights_initializer=tf.random_normal_initializer(), scope='fc4',
            #                                          activation_fn=tf.nn.tanh)
            # fc_v2 = tf.contrib.layers.fully_connected(fc_v, 7, biases_initializer=tf.random_normal_initializer(),
            #                                           weights_initializer=tf.random_normal_initializer(), scope='fc5',
            #                                           activation_fn=tf.nn.tanh)
            final_vec = tf.concat(1, [fc3, action_vec])
            connect_fc = tf.contrib.layers.fully_connected(final_vec, 7,
                                                           biases_initializer=tf.random_normal_initializer(), scope='fc6',
                                                           weights_initializer=tf.random_normal_initializer(),
                                                           activation_fn=tf.nn.tanh)
            final = tf.reshape(connect_fc, [-1, 7, 1])
            return final



    def net_try16(self, board, action_vec):
        print("NETWORK 3")
        conv_layer1 = tf.contrib.layers.conv2d(board, 8, [5, 5], padding='SAME', activation_fn=tf.nn.tanh)
        conv_layer2 = tf.contrib.layers.conv2d(conv_layer1, 16, [5, 5], padding='SAME', activation_fn=tf.nn.tanh)
        conv_layer3 = tf.contrib.layers.conv2d(conv_layer2, 32, [3, 3], padding='SAME', activation_fn=tf.nn.tanh)
        conv_layer4 = tf.contrib.layers.conv2d(conv_layer3, 1, [3, 3], padding='SAME', activation_fn=tf.nn.tanh)

        out_shape = conv_layer4.get_shape().as_list()
        flat_out = tf.reshape(conv_layer4, [-1, out_shape[1] * out_shape[2]])
        fc = tf.contrib.layers.fully_connected(flat_out, 14, tf.nn.tanh, weights_initializer=tf.random_normal_initializer(),
                                               biases_initializer=tf.random_normal_initializer())
        fc = tf.reshape(fc, [-1, 14])
        action_vec = tf.reshape(action_vec, [-1, 14])

        together = tf.concat(1, [fc, action_vec])
        connect_fc = tf.contrib.layers.fully_connected(together, 7, biases_initializer=tf.random_normal_initializer(),
                                                       scope='fc6', weights_initializer=tf.random_normal_initializer(),
                                                       activation_fn=tf.nn.tanh)
        final = tf.reshape(connect_fc, [-1, 7, 1])
        return final


    def net_try17(self, board, action_vec):
        print("NETWORK 17")
        conv_layer1 = tf.contrib.layers.conv2d(board, 32, [5, 5], padding='SAME', activation_fn=tf.nn.sigmoid)
        conv_layer2 = tf.contrib.layers.conv2d(conv_layer1, 16, [5, 5], padding='SAME', activation_fn=tf.nn.sigmoid)
        conv_layer3 = tf.contrib.layers.conv2d(conv_layer2, 16, [5, 5], padding='SAME', activation_fn=tf.nn.sigmoid)
        conv_layer4 = tf.contrib.layers.conv2d(conv_layer3, 1, [5, 5], padding='SAME', activation_fn=tf.nn.sigmoid)

        out_shape = conv_layer4.get_shape().as_list()
        flat_out = tf.reshape(conv_layer4, [-1, out_shape[1] * out_shape[2]])
        fc = tf.contrib.layers.fully_connected(flat_out, 14, tf.nn.tanh, weights_initializer=tf.random_normal_initializer(),
                                               biases_initializer=tf.random_normal_initializer())
        fc = tf.reshape(fc, [-1, 14])
        action_vec = tf.reshape(action_vec, [-1, 14])

        together = tf.concat(1, [fc, action_vec])
        connect_fc = tf.contrib.layers.fully_connected(together, 7, biases_initializer=tf.random_normal_initializer(),
                                                       scope='fc6', weights_initializer=tf.random_normal_initializer(),
                                                       activation_fn=tf.nn.tanh)
        final = tf.reshape(connect_fc, [-1, 7, 1])
        return final


def printShape(tensor):
    print(tensor.get_shape().as_list())


