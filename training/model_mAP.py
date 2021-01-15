import numpy as np
import os
from absl import app, flags, logging
from absl.flags import FLAGS
from benchmark import calc_map
flags.DEFINE_string('train_dataset', './data/particle_train.tfrecord', 'path to dataset')
flags.DEFINE_string('val_dataset', './data/particle_val.tfrecord', 'path to validation dataset')
flags.DEFINE_string('weights_path', './checkpoints/', 'path to weights files')
flags.DEFINE_string('classes', './data/particle.names', 'path to classes file')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_integer('num_classes', 1, 'number of classes in the model')
flags.DEFINE_integer('epochs', 1, 'ending epochs to calcuate mAP')
flags.DEFINE_integer('epochs_start', 1, 'starting epochs to calcuate mAP')

# train = './data/particle_train.tfrecord'
# val = './data/particle_val.tfrecord'

# classes = './data/particle.names'
# num_classes = 1

# model = './checkpoints/'

# num = 25

def main(_argv):
	train_mAP = []
	val_mAP = []

	with open("mAP.txt", "w") as f:
		for i in range(FLAGS.epochs_start, FLAGS.epochs + 1):
			weights_path = FLAGS.weights_path + 'yolov3_train_{}.tf'.format(i)
			# print(model_path)
			os.system("python detect_list.py --classes {} --size {} --weights {} --num_classes {} --tfrecord {} --batch_size {}".format(FLAGS.classes, FLAGS.size, weights_path, FLAGS.num_classes, FLAGS.train_dataset, FLAGS.batch_size))
			# detect_tfrecord(record_path = train, weights = weights_path, classes = classes, num_classes = num_classes)

			train_mAP.append(calc_map())
			print(train_mAP[-1])

			# detect_tfrecord(record_path = val, weights = weights_path, classes = classes, num_classes = num_classes)
			os.system("python detect_list.py --classes {} --size {} --weights {} --num_classes {} --tfrecord {} --batch_size {}".format(FLAGS.classes, FLAGS.size, weights_path, FLAGS.num_classes, FLAGS.val_dataset, FLAGS.batch_size))
			
			val_mAP.append(calc_map())
			print(val_mAP[-1])

			f.write("{}, {}\n".format(train_mAP[-1], val_mAP[-1]))

			

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

