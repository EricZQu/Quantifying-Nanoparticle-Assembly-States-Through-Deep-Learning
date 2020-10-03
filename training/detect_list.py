import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
import os, glob

#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags.DEFINE_string('classes', './data/particle.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('tfrecord', None, 'path to tfrecord')
flags.DEFINE_integer('num_classes', 1, 'number of classes in the model')
flags.DEFINE_integer('batch_size', 12, 'number of batch size for detection')

def DeleteCache():
    for filePath in glob.glob("benchmark/ground-truth/*.txt"):
        os.remove(filePath)
    for filePath in glob.glob("benchmark/detection-results/*.txt"):
        os.remove(filePath)

def detect_tfrecord_batch(record_path, weights = './checkpoints/yolov3.tf', classes = './data/particle.names', num_classes = 1, size = 416, batch_size = 1):
    DeleteCache()

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(weights).expect_partial()
    logging.info('weights loaded: ' + weights)

    class_names = [c.strip() for c in open(classes).readlines()]
    logging.info('classes loaded: ' + classes)

    dataset = load_tfrecord_dataset(record_path, classes, size)
    dataset = dataset.shuffle(512)
    tot1 = 0
    tot2 = 0
    times = []
    imgs = []
    for img_raw, label_raw in iter(dataset):
        tot1 += 1
        label_raw = label_raw.numpy()
        labels = []
        for row in label_raw:
            if np.sum(row) > 0:
                labels.append(row)
        lables = np.asarray(labels)
        with open('benchmark/ground-truth/{}.txt'.format(tot1), "w") as f:
            for boxes in labels:
                f.write("praticle ")
                for j in range(0, 4):
                    f.write(str(boxes[j] * size) + " ")
                f.write("\n")
        imgs.append(img_raw)

        if tot1 % batch_size == 0:
            imgs = transform_images(imgs, size, random = False)
            t1 = time.time()
            boxes, scores, classes, nums = yolo.predict(imgs)
            t2 = time.time()
            times.append(t2 - t1)

            for k in range(len(nums)):
                tot2 += 1
                with open('benchmark/detection-results/{}.txt'.format(tot2), "w") as f:
                    for i in range(nums[k]):
                        f.write("praticle ")
                        f.write(str(scores[k][i]) + " ")
                        for j in range(0, 4):
                            f.write(str(boxes[k][i][j] * size) + " ")
                        f.write("\n")
            imgs = []
        # break
    if len(imgs) != 0:
        imgs = transform_images(imgs, size, random = False)
        boxes, scores, classes, nums = yolo.predict(imgs)
        for k in range(len(nums)):
            tot2 += 1
            with open('benchmark/detection-results/{}.txt'.format(tot2), "w") as f:
                for i in range(nums[k]):
                    f.write("praticle ")
                    f.write(str(scores[k][i]) + " ")
                    for j in range(0, 4):
                        f.write(str(boxes[k][i][j] * size) + " ")
                    f.write("\n")
    logging.info("total number of images: {}".format(tot1))
    logging.info('Time: {:.2f} ms'.format(np.average(np.asarray(times)) * 1000))


def main(_argv):
    detect_tfrecord_batch(record_path = FLAGS.tfrecord, weights = FLAGS.weights, classes = FLAGS.classes, num_classes = FLAGS.num_classes, batch_size = FLAGS.batch_size)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
