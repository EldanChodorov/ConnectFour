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
        self.boards = tf.placeholder(tf.float32, shape=[None, 6, 7, 1], name="boards")
        self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
        self.q_vals = self.network_run(self.boards)
        self.action = tf.reduce_max(self.q_vals * self.actions_holder,axis=1)
        self.punishment = tf.placeholder(tf.float32, shape=[None], name="punishment")
        self.loss = tf.reduce_mean(tf.pow(self.rewards - self.action, 2) * self.punishment)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.init = tf.initialize_all_variables()
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
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
        return self.net_try2(feed_inputs)

    def net_try1(self, inputs):
        conv_layer1 = tf.layers.conv2d(inputs, 16, [5, 5], padding='same', activation=tf.nn.relu,
                                               name="conv1")
        conv_layer2 = tf.layers.conv2d(conv_layer1, 32, [3, 3], padding='same',
                                               activation=tf.nn.relu, name="conv2")
        conv_layer3 = tf.layers.conv2d(conv_layer2, 64, [3, 3], padding='same',
                                               activation=tf.nn.relu, name="conv3")
        sum_layer1 = tf.reduce_sum(conv_layer3, reduction_indices=1)
        fully_connected1 = tf.layers.dense(sum_layer1, 1, activation=None)
        return fully_connected1

    def net_try2(self, inputs):
        W_conv1 = self.weight([3, 3, 1, 8])
        b_conv1 = self.bias([8])

        h_conv1 = tf.nn.relu(tf.nn.conv2d(inputs, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        h_pool1 = tf.nn.dropout(h_pool1,keep_prob=0.7)
        W_conv2 = self.weight([3, 3, 8, 16])
        b_conv2 = self.bias([16])

        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        h_pool_shape = h_pool2.get_shape().as_list()

        W_fc1 = self.weight([h_pool_shape[1] * h_pool_shape[2] * h_pool_shape[3], 10])
        b_fc1 = self.bias([10])

        h_pool2_flat = tf.reshape(h_pool2, [-1, h_pool_shape[1] * h_pool_shape[2] * h_pool_shape[3]])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        W_fc2 = self.weight([10, 7])
        b_fc2 = self.bias([7])

        final = tf.matmul(h_fc1, W_fc2) + b_fc2
        final = tf.expand_dims(final, -1)
        return final


    def net_try3(self, inputs):
        conv_layer1 = tf.contrib.layers.conv2d(inputs, 8, [5, 5], padding='same', activation=tf.nn.relu, name="conv1")
        conv_layer2 = tf.contrib.layers.conv2d(conv_layer1, 16, [5, 5], padding='same', activation=tf.nn.relu, name="conv2")
        conv_layer3 = tf.contrib.layers.conv2d(conv_layer2, 32, [3, 3], padding='same', activation=tf.nn.relu, name="conv3")
        conv_layer4 = tf.contrib.layers.conv2d(conv_layer3, 64, [3, 3], padding='same', activation=tf.nn.relu, name="conv4")
        sum_layer1 = tf.reduce_sum(conv_layer4, reduction_indices=1)
        fully_connected1 = tf.contrib.layers.fully_connected(sum_layer1, 1, activation_fn=None)
        return fully_connected1

    def net_try4(self, inputs):

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

    def net_try5(self, inputs):

        input_shape = inputs.get_shape().as_list()

        fully_connected1 = tf.layers.dense(inputs, 4 )
        conv_layer1 = tf.layers.conv2d(fully_connected1, 4, [5, 5],2, padding='same', activation=tf.nn.relu,
                                               name="conv1")
        fully_connected2 = tf.layers.dense(conv_layer1, 8)
        conv_layer3 = tf.layers.conv2d(fully_connected2, 8, [3, 3], padding='same',
                                               activation=tf.nn.relu,
                                               name="conv3")
        fully_connected3 = tf.layers.dense(conv_layer3, 16)
        conv_layer4 = tf.layers.conv2d(fully_connected3, 16, [3, 3], padding='same',
                                               activation=tf.nn.relu,
                                               name="conv4")
        conv_layer5 = tf.layers.conv2d_transpose(conv_layer4,4,[3,3],2)
        sum_layer1 = tf.reduce_sum(conv_layer5, reduction_indices=1)
        fully_connected4 = tf.layers.dense(sum_layer1, 1, activation=None)
        return fully_connected4


if __name__ == '__main__':
    matrix = np.zeros((6, 7)).astype(np.float32)
    matrix[([5,4,5],[2,2,3])] = 1
    matrix[([5,4,5],[4,4,1])] = 2
    matrix = matrix.reshape((6, 7, 1))
    matrix = matrix[None, :, :, :]
    batch_size = 50
    reward = np.array([0,1,1,-1,1,0,1])
    action = np.array([1] * 7)
    action = action[None, :]
    network = PolicyNetwork(0.1, 2, batch_size)
    network.train(matrix, reward, action)
