import tensorflow as tf
import os
import random
import sys
import pandas as pd
import numpy as np


def get_data(dataset):
    print("Loading dataset...")
    table = pd.read_csv("../../data/"+dataset, header=None)
    filenames = [item[0].split("/")[-1].split('.')[0] for item in table.values]
    class_ids = [int(item[0].split("/")[-1].split("_")[-1].split(".")[0])-1 for item in table.values]
    data = []
    for index, filename in enumerate(filenames):
        visit = np.load("../../data/npy/train_visit2/"+filename+".npy")
        label = class_ids[index]
        data.append([visit, label])
    random.seed(0)
    random.shuffle(data)
    print("Loading completed...")
    return data


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(data, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'visit': bytes_feature(data),
        'label': int64_feature(label),
    }))


def _convert_dataset(data, tfrecord_path, dataset):
    """ Convert data to TFRecord format. """
    output_filename = os.path.join(tfrecord_path, dataset+"_visit2.tfrecord")
    tfrecord_writer = tf.python_io.TFRecordWriter(output_filename)
    length = len(data)
    for index, item in enumerate(data):
        data_ = item[0].tobytes()
        label = item[1]
        example = image_to_tfexample(data_, label)
        tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\r>> Converting image %d/%d' % (index + 1, length))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()


if __name__ == '__main__':
    data = get_data("train_oversampling.txt")
    _convert_dataset(data, "../../data/tfrecord/", "train")
    data = get_data("valid.txt")
    _convert_dataset(data, "../../data/tfrecord/", "valid")
