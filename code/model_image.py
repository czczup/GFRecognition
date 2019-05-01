import tensorflow as tf
from functools import reduce
from operator import mul


class Resnet(object):
    def __init__(self):
        with tf.name_scope("image"):
            self.image = tf.placeholder(tf.float32, [None, 88, 88, 3], name='image')
        with tf.name_scope("label"):
            self.label = tf.placeholder(tf.int32, [None], name='label')
            self.one_hot = tf.one_hot(indices=self.label, depth=9, name='one_hot')

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.training = tf.placeholder(tf.bool)
        self.output = self.model(self.image)
        self.loss = self.get_loss(self.output, self.one_hot)
        self.batch_size = 512
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.one_hot, 1))
        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('train_accuracy', self.accuracy)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(1e-3).minimize(self.loss, global_step=self.global_step)

        self.merged = tf.summary.merge_all()
        print("网络初始化成功")

    def conv2d(self, x, input_filters, output_filters, kernel, strides=1, padding="SAME"):
        with tf.name_scope('conv'):
            shape = [kernel, kernel, input_filters, output_filters]
            weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
            return tf.nn.conv2d(x, weight, strides=[1, strides, strides, 1], padding=padding, name='conv')

    def residual(self, x, num_filters, strides, with_shortcut=False):
        with tf.name_scope('residual'):
            conv1 = self.conv2d(x, num_filters[0], num_filters[1], kernel=1, strides=strides)
            bn1 = tf.layers.batch_normalization(conv1, axis=3, training=self.training)
            relu1 = tf.nn.relu(bn1)
            conv2 = self.conv2d(relu1, num_filters[1], num_filters[2], kernel=3)
            bn2 = tf.layers.batch_normalization(conv2, axis=3, training=self.training)
            relu2 = tf.nn.relu(bn2)
            conv3 = self.conv2d(relu2, num_filters[2], num_filters[3], kernel=1)
            bn3 = tf.layers.batch_normalization(conv3, axis=3, training=self.training)
            if with_shortcut:
                shortcut = self.conv2d(x, num_filters[0], num_filters[3], kernel=1, strides=strides)
                bn_shortcut = tf.layers.batch_normalization(shortcut, axis=3, training=self.training)
                residual = tf.nn.relu(bn_shortcut+bn3)
            else:
                residual = tf.nn.relu(x+bn3)
        return residual

    def model(self, image):
        channel = 16
        with tf.name_scope("Resnet"):
            with tf.name_scope('stage1'):
                conv = self.conv2d(image, 3, channel, 7, 1)
                bn = tf.layers.batch_normalization(conv, axis=3, training=self.training)
                relu = tf.nn.relu(bn)
                print(relu)
            with tf.name_scope('stage2'):
                pool = tf.nn.max_pool(relu, [1, 3, 3, 1], [1, 2, 2, 1], padding="SAME")
                res = self.residual(pool, [channel, channel//2, channel//2, channel*2], 1, with_shortcut=True)
                # res = self.residual(res, [channel*2, channel//2, channel//2, channel*2], 1)
                # res = self.residual(res, [channel*2, channel//2, channel//2, channel*2], 1)
                print(res)
            with tf.name_scope('stage3'):
                res = self.residual(res, [channel*2, channel, channel, channel*4], 2, with_shortcut=True)
                # res = self.residual(res, [channel*4, channel, channel, channel*4], 1)
                # res = self.residual(res, [channel*4, channel, channel, channel*4], 1)
                # res = self.residual(res, [channel*4, channel, channel, channel*4], 1)
                print(res)
            with tf.name_scope('stage4'):
                res = self.residual(res, [channel*4, channel*2, channel*2, channel*8], 2, with_shortcut=True)
                # res = self.residual(res, [channel*8, channel*2, channel*2, channel*8], 1)
                # res = self.residual(res, [channel*8, channel*2, channel*2, channel*8], 1)
                # res = self.residual(res, [channel*8, channel*2, channel*2, channel*8], 1)
                # res = self.residual(res, [channel*8, channel*2, channel*2, channel*8], 1)
                # res = self.residual(res, [channel*8, channel*2, channel*2, channel*8], 1)
                print(res)
            with tf.name_scope('stage5'):
                res = self.residual(res, [channel*8, channel*4, channel*4, channel*16], 2, with_shortcut=True)
                # res = self.residual(res, [channel*16, channel*4, channel*4, channel*16], 1)
                # res = self.residual(res, [channel*16, channel*4, channel*4, channel*16], 1)
                print(res)
                pool = tf.nn.avg_pool(res, [1, 6, 6, 1], strides=[1, 1, 1, 1], padding='VALID')
                print(pool)
            with tf.name_scope('fc'):
                flatten = tf.layers.flatten(pool)
                print(flatten)
                output = tf.layers.dense(flatten, units=9)
        return output

    def get_loss(self, output, onehot):
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=onehot)
            loss = tf.reduce_mean(losses)
            tf.summary.scalar('loss', loss)
        return loss

    def get_num_params(self):
        num_params = 0
        for variable in tf.trainable_variables():
            print(variable)
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        return num_params


if __name__ == '__main__':
    model = Resnet()
    print(model.get_num_params())
