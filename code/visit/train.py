from model import VisitModel
import tensorflow as tf
import time
import os


def read_and_decode_train(filename):
    filename_queue = tf.train.string_input_producer([filename])  # create a queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # return file_name and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'visit': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                       })  # return image and label
    image = tf.decode_raw(features['visit'], tf.float64)
    image = tf.reshape(image, [7, 26, 24])
    label = tf.cast(features['label'], tf.int64)
    return image, label


def read_and_decode_valid(filename):
    filename_queue = tf.train.string_input_producer([filename])  # create a queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # return file_name and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'visit': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                       })  # return image and label
    image = tf.decode_raw(features['visit'], tf.float64)
    image = tf.reshape(image, [7, 26, 24])
    label = tf.cast(features['label'], tf.int64)
    return image, label


def load_training_set():
    with tf.name_scope('input_train'):
        image_train, label_train = read_and_decode_train("../../data/tfrecord/train_visit.tfrecord")
        image_batch_train, label_batch_train = tf.train.shuffle_batch(
            [image_train, label_train], batch_size=batch_size, capacity=10240, min_after_dequeue=5120, num_threads=4
        )
    return image_batch_train, label_batch_train


def load_valid_set():
    with tf.name_scope('input_valid'):
        image_valid, label_valid = read_and_decode_valid("../../data/tfrecord/valid_visit.tfrecord")
        image_batch_valid, label_batch_valid = tf.train.shuffle_batch(
            [image_valid, label_valid], batch_size=batch_size, capacity=10240, min_after_dequeue=5120, num_threads=4
        )
    return image_batch_valid, label_batch_valid


def train(model):
    amount = 84078
    image_batch_train, label_batch_train = load_training_set()
    image_batch_valid, label_batch_valid = load_valid_set()

    # Adaptive use of GPU memory.
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        # general setting
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        # Recording training process.
        writer_train = tf.summary.FileWriter("../../model/"+dirId+"/log/train", sess.graph)
        writer_valid = tf.summary.FileWriter("../../model/"+dirId+"/log/valid", sess.graph)

        last_file = tf.train.latest_checkpoint("../../model/"+dirId)
        var_list = [var for var in tf.global_variables() if "moving" in var.name]
        var_list += [var for var in tf.global_variables() if "global_step" in var.name]
        var_list += tf.trainable_variables()
        saver = tf.train.Saver(var_list=var_list, max_to_keep=20)
        if last_file:
            tf.logging.info('Restoring model from {}'.format(last_file))
            saver.restore(sess, last_file)
        # train
        while True:
            time1 = time.time()
            image_train, label_train, step = sess.run(
                [image_batch_train, label_batch_train, model.global_step])
            _, loss_ = sess.run([model.optimizer, model.loss], feed_dict={model.image: image_train,
                                                                          model.label: label_train,
                                                                          model.training: True})
            print('[epoch %d, step %d/%d]: loss %.6f' % (
            step // (amount // batch_size), step % (amount // batch_size), amount // batch_size, loss_),
                  'time %.3fs' % (time.time() - time1))
            if step % 10 == 0:
                image_train, label_train = sess.run([image_batch_train, label_batch_train])
                acc_train, summary = sess.run([model.accuracy, model.merged], feed_dict={model.image: image_train,
                                                                                         model.label: label_train,
                                                                                         model.training: True})
                writer_train.add_summary(summary, step)
                image_valid, label_valid = sess.run([image_batch_valid, label_batch_valid])
                acc_valid, summary, output = sess.run([model.accuracy, model.merged, model.output], feed_dict={model.image: image_valid,
                                                                                         model.label: label_valid,
                                                                                         model.training: True})
                print(output[0])
                writer_valid.add_summary(summary, step)
                print('[epoch %d, step %d/%d]: train acc %.3f, valid acc %.3f' % (step // (amount // batch_size),
                                                                                  step % (amount // batch_size),
                                                                                  amount // batch_size, acc_train,
                                                                                  acc_valid),
                      'time %.3fs' % (time.time() - time1))
            if step % 100 == 0:
                print("Save the model Successfully")
                saver.save(sess, "../../model/"+dirId+"/model.ckpt", global_step=step)

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    deviceId = input("please input device id (0-7): ")
    os.environ["CUDA_VISIBLE_DEVICES"] = deviceId
    dirId = input("please input dir id: ")
    model = VisitModel()
    batch_size = model.batch_size
    train(model)
