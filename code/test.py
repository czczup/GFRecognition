from model_visit import VisitModel
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import os




def test():
    table = pd.read_csv("../valid.txt", header=None)
    filenames = [item[0].split('/')[-1].split('.')[0] for item in table.values]

    model = VisitModel()
    var_list = [var for var in tf.global_variables() if "moving" in var.name]
    var_list += [var for var in tf.global_variables() if "global_step" in var.name]
    var_list += tf.trainable_variables()
    saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "../model/1012/model.ckpt-500")

    predictions = {}
    length = len(filenames)
    for index, filename in enumerate(filenames):
        array = np.load("../data/train_visit/"+filename+".npy")
        class_id = int(filename.split('_')[-1])

        prediction = sess.run(tf.argmax(model.output, 1),
                              feed_dict={model.input: [array],
                                         model.training: False})[0]
        print(prediction, class_id)

        sys.stdout.write('\r>> Testing image %d/%d'%(index+1, length))
        sys.stdout.flush()


if __name__=='__main__':
    deviceId = input("please input device id (0-7): ")
    os.environ["CUDA_VISIBLE_DEVICES"] = deviceId
    test()
