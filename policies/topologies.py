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

        # placeholders
        self.actions_holder = tf.placeholder(tf.float32, shape=[None, 7, 1], name="action_holder")
        self.boards = tf.placeholder(tf.float32, shape=[None, 5, 6, 7], name="boards")
        self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
        self.q_vals = self.network_run(self.boards)

        self.action = tf.reduce_max(self.q_vals*self.actions_holder, reduction_indices=1)
        self.punishment = tf.placeholder(tf.float32, shape=[None], name="punishment")
        self.loss = tf.reduce_mean(tf.pow(self.rewards - self.action, 2) * self.punishment)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.probabilities = tf.nn.softmax(tf.squeeze(self.q_vals, [-1]))

        self.init = tf.initialize_all_variables()
        self.session = tf.Session()
        self.session.run(self.init)

    def train(self, inputs, rewards, actions):
        invalid_move = 1.0
        for ep in range(self.epochs):
            for batch in range(self.batches_per_epoch):
                _, loss, q_values = self.session.run([self.optimizer, self.loss, self.q_vals],
                                                feed_dict={self.rewards: rewards, self.actions_holder:
                                                    actions, self.boards: inputs, self.punishment: invalid_move})
                invalid_move = 1.0 + (inputs[0, np.argmax(q_values, axis=1)] != 0) * 0.1

    def weight(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def bias(self, shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    def network_run(self, feed_inputs):
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



        h_conv1 = tf.layers.conv2d(inputs,8, [5,5],activation=tf.nn.relu, padding='SAME',
                                   kernel_initializer=tf.random_normal_initializer(),
           bias_initializer=tf.random_normal_initializer())
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',kernel_initializer=tf.random_normal_initializer(),
           bias_initializer=tf.random_normal_initializer())

        h_conv2 = tf.layers.conv2d(h_pool1,16,[5,5],padding='SAME',activation=tf.nn.relu,kernel_initializer=tf.random_normal_initializer(),
           bias_initializer=tf.random_normal_initializer())
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',kernel_initializer=tf.random_normal_initializer(),
           bias_initializer=tf.random_normal_initializer())

        # h_pool_shape = h_pool2.get_shape().as_list()
        fc3 = tf.contrib.layers.fully_connected(h_pool2, 16,
                                                biases_initializer=tf.random_normal_initializer(),
                                                weights_initializer=tf.random_normal_initializer())
        h_pool3 = tf.nn.max_pool(fc3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        fc4 = tf.contrib.layers.fully_connected(h_pool3, 7,
                                                biases_initializer=tf.random_normal_initializer(),
                                                weights_initializer=tf.random_normal_initializer())

        final = tf.reshape(fc4, [-1,7,1])
        return final


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
        outer_lyears= tf.reduce_max(outer_lyears,axis=2)
        outer_lyears = tf.reduce_max(outer_lyears,axis=1)
        outer_lyears_shape = outer_lyears.get_shape().as_list()
        flatten_outer = tf.reshape(outer_lyears, [-1, outer_lyears_shape[1] * outer_lyears_shape[2]])
        flatten_inner = tf.reshape(first_stat, [-1, first_stat_shape[1] * first_stat_shape[2]])
        total_flatten = tf.concat([flatten_inner,flatten_outer],axis=1)
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

        first_stat = inputs[:,0,:,:]
        first_stat_shape = first_stat.get_shape().as_list()
        outer_lyears = inputs[:,1:,:,:]
        outer_lyears= tf.reduce_max(outer_lyears,axis=2)
        outer_lyears = tf.reduce_max(outer_lyears,axis=1)
        flatten_inner = tf.reshape(first_stat, [-1, first_stat_shape[1] * first_stat_shape[2]])
        fc1 = tf.contrib.layers.fully_connected(flatten_inner, 32,biases_initializer=tf.random_normal_initializer(),
                                                weights_initializer=tf.random_normal_initializer())
        fc2 = tf.contrib.layers.fully_connected(fc1, 16, biases_initializer=tf.random_normal_initializer(),
                                                weights_initializer=tf.random_normal_initializer())
        fc3 = tf.contrib.layers.fully_connected(fc2, 7, biases_initializer=tf.random_normal_initializer(),
                                                weights_initializer=tf.random_normal_initializer(),)

        fc_v = tf.contrib.layers.fully_connected(outer_lyears, 7,biases_initializer=tf.random_normal_initializer(),
                                                weights_initializer=tf.random_normal_initializer(),
                               activation_fn=tf.nn.sigmoid)
        connect = tf.multiply(fc3,fc_v)
        connect_fc = tf.contrib.layers.fully_connected(connect, 7,
                                                 biases_initializer=tf.random_normal_initializer(),
                                          weights_initializer=tf.random_normal_initializer(),activation_fn=None)
        final = tf.reshape(connect_fc, [-1, 7, 1])
        return final
