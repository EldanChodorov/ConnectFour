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
        self.actions_holder = tf.placeholder(tf.float32, shape=[None, 7])
        self.boards = tf.placeholder(tf.float32, shape=[None, 6, 7, 1])
        self.rewards = tf.placeholder(tf.float32, shape=[None])
        self.x = self.network_run(self.boards)

        self.action = tf.reduce_sum(tf.mul(self.x, self.actions_holder), reduction_indices=1)

        self.loss = tf.reduce_mean(tf.pow(self.rewards - self.action, 2))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(self.init)

    def train(self, inputs, rewards, actions):
        for ep in range(self.epochs):
            for batch in range(self.batches_per_epoch):
                _, loss, net = self.session.run([self.optimizer, self.loss, self.x],
                                                feed_dict={self.rewards: rewards, self.actions_holder:
                                                    actions, self.boards: inputs})
                if batch % 20 == 0:
                    print(loss)
                    print(net)

    def weight(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def bias(self, shape):
        return tf.Variable(tf.constant(0.1, shape))

    def network_run(self, feed_inputs):
        '''
        Runs the given input through the network architecture.
        Architecture:
        ...
        :param feed_inputs: matrix [numpy.ndarray] 6x7
        :return: result from network. - [1X7]
        '''
        self.x = feed_inputs
        lays = tf.contrib.layers
        conv_layer1 = lays.conv2d(self.x, 8, [5, 5], activation_fn=tf.nn.sigmoid)
        conv_layer2 = lays.conv2d(conv_layer1, 16, [3, 3], activation_fn=tf.nn.sigmoid)
        conv_layer3 = lays.conv2d(conv_layer2, 8, [3, 3], activation_fn=tf.nn.sigmoid)
        sum_layer1 = tf.reduce_sum(conv_layer3, axis=1)
        fully_connected1 = lays.fully_connected(sum_layer1, 1, activation_fn=None)

        return fully_connected1

    def __del__(self):
        if self.session:
            self.session.close()
            self.session = None


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
