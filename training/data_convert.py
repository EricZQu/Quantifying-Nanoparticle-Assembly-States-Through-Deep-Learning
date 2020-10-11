import time
import os
import hashlib

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import numpy as np


flags.DEFINE_string('dataset', '', 'path to dataset label file')
flags.DEFINE_string('output', 'data/', 'path to output folder')
flags.DEFINE_float('val_split', 0.2, 'validation split')
flags.DEFINE_float('subset', 1, 'get a subset of training set (0 ~ 1)')

def build_example(boxes):
    img_path = boxes[0]
    img_raw = open(img_path, 'rb').read()
    key = hashlib.sha256(img_raw).hexdigest()

    width = 416
    height = 416

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []

    for i in range(1, len(boxes)):
        s = boxes[i]
        bndbox = (s.split(','))
        xmin.append(float(bndbox[0]) / width)
        ymin.append(float(bndbox[1]) / height)
        xmax.append(float(bndbox[2]) / width)
        ymax.append(float(bndbox[3]) / height)
        classes_text.append("particle".encode('utf8'))
        classes.append(0)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[boxes[0].encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[boxes[0].encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['png'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
    }))
    return example


def main(_argv):

    with open(FLAGS.dataset, "r") as f:
        lines = f.readlines()

    np.random.shuffle(lines)
    lines = lines[:int(len(lines) * FLAGS.subset)]
    val_split = FLAGS.val_split
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    train_lines = lines[:num_train]
    val_lines = lines[num_train:]

    writer = tf.io.TFRecordWriter(FLAGS.output + "particle_train.tfrecord")
    for line in train_lines:
        boxes = line.split()
        tf_example = build_example(boxes)
        writer.write(tf_example.SerializeToString())
    writer.close()

    writer = tf.io.TFRecordWriter(FLAGS.output + "particle_val.tfrecord")
    for line in val_lines:
        boxes = line.split()
        tf_example = build_example(boxes)
        writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    app.run(main)
