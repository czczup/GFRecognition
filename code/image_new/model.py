import tensorflow as tf
from functools import reduce
from operator import mul
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np


class SE_ResNeXt(object):
    def __init__(self):
        self.image = tf.placeholder(tf.float32, [None, 88, 88, 3], name='image')
        self.label = tf.placeholder(tf.int32, [None], name='label')
        self.one_hot = tf.one_hot(indices=self.label, depth=9, name='one_hot')
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.is_training = tf.placeholder(tf.bool)

        self.out_dims = [8, 16, 32, 64, 128]
        self.cardinality = 8
        self.num_block = 1
        self.depth = 8
        self.reduction_ratio = 4

        self.output = self.build_SEnet(self.image)
        self.loss = self.get_loss(self.output, self.one_hot)
        self.batch_size = 256

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


    def batch_norm(self, x, is_training, scope):
        with arg_scope(
                [batch_norm],
                scope=scope,
                updates_collections=None,
                decay=0.9,
                center=True,
                scale=True,
                zero_debias_moving_mean=True):
            return tf.cond(
                is_training,
                lambda: batch_norm(inputs=x, is_training=is_training, reuse=None),
                lambda: batch_norm(inputs=x, is_training=is_training, reuse=True))

    def conv_bn_layer(self,
                      x,
                      filters,
                      filter_size,
                      stride,
                      scope,
                      padding="same"):
        with tf.name_scope(scope):
            x = tf.layers.conv2d(
                inputs=x,
                filters=filters,
                kernel_size=filter_size,
                strides=stride,
                padding=padding,
                use_bias=False)
            x = self.batch_norm(
                x, is_training=self.is_training, scope=scope+"_batch1")
            return tf.nn.relu(x)

    def transform_layer(self, x, stride, depth, scope):
        x = self.conv_bn_layer(
            x,
            filters=depth,
            filter_size=1,
            stride=1,
            padding="same",
            scope=scope+"_trans1")
        return self.conv_bn_layer(
            x,
            filters=depth,
            filter_size=3,
            stride=stride,
            padding="same",
            scope=scope+"_trans2")

    def split_layer(self, input_x, stride, depth, layer_name, cardinality):
        with tf.name_scope(layer_name):
            layer_splits = []
            for i in range(cardinality):
                layer_splits.append(
                    self.transform_layer(input_x, stride, depth,
                                         layer_name+"_splitN_"+str(i)))
            return tf.concat(layer_splits, axis=3)  # concatenate along channel

    def transition_layer(self, x, out_dim, scope):
        """A 1 x 1 convolution.
        """

        return self.conv_bn_layer(
            x,
            filters=out_dim,
            filter_size=1,
            stride=1,
            padding="same",
            scope=scope)

    def squeeze_excitation_layer(self, input_x, out_dim, reduction_ratio,
                                 layer_name):
        with tf.name_scope(layer_name):
            pool = global_avg_pool(input_x)
            squeeze = tf.layers.dense(
                pool,
                use_bias=False,
                units=out_dim/reduction_ratio,
            )
            squeeze = tf.nn.relu(squeeze)
            excitation = tf.layers.dense(
                squeeze, units=out_dim, use_bias=False)
            excitation = tf.nn.sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
            return input_x*excitation

    def residual_layer(self, input_x, out_dim, layer_num, cardinality, depth,
                       reduction_ratio, num_block):
        for i in range(num_block):
            # The input here must follow channel last format
            input_dim = int(np.shape(input_x)[-1])

            if input_dim*2==out_dim:
                flag = True
                stride = 2
                channel = input_dim//2
            else:
                flag = False
                stride = 1

            x = self.split_layer(
                input_x,
                stride=stride,
                cardinality=cardinality,
                depth=depth,
                layer_name="split_layer_"+layer_num+"_"+str(i))
            x = self.transition_layer(
                x,
                out_dim=out_dim,
                scope="trans_layer_"+layer_num+"_"+str(i))
            x = self.squeeze_excitation_layer(
                x,
                out_dim=out_dim,
                reduction_ratio=reduction_ratio,
                layer_name="squeeze_layer_"+layer_num+"_"+str(i))

            if flag is True:
                pad_input_x = tf.layers.average_pooling2d(
                    input_x, pool_size=[2, 2], strides=2, padding="same")
                pad_input_x = tf.pad(
                    pad_input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]])
            else:
                pad_input_x = input_x
            input_x = tf.nn.relu(x+pad_input_x)
        return input_x

    def build_SEnet(self, input_x):
        input_x = self.conv_bn_layer(
            input_x,
            filters=self.out_dims[0],
            filter_size=3,
            stride=1,
            scope="first_layer")
        print(input_x)

        for i, out_dim in enumerate(self.out_dims[1:]):
            x = self.residual_layer(
                (x if i else input_x),
                out_dim=out_dim,
                num_block=self.num_block,
                depth=self.depth,
                cardinality=self.cardinality,
                reduction_ratio=self.reduction_ratio,
                layer_num=str(i+1))
            print(x)

        x = global_avg_pool(x)
        print(x)
        x = flatten(x)
        print(x)
        return tf.layers.dense(inputs=x, use_bias=False, units=9)

    def get_loss(self, output, onehot):
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=onehot)
            loss = tf.reduce_mean(losses)
            tf.summary.scalar('loss', loss)
        return loss

    def get_num_params(self):
        num_params = 0
        for variable in tf.trainable_variables():
            # print(variable)
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        return num_params


if __name__=='__main__':
    model = SE_ResNeXt()
    print(model.get_num_params())
