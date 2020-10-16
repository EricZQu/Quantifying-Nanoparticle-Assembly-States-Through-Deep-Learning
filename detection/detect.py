from PIL import Image
import os, glob
import hashlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import KDTree
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

flags.DEFINE_string('classes', './yolov3_tf2/particle.names', 'path to classes file')
flags.DEFINE_string('weights', './yolov3_tf2/yolov3_model.tf', 'path to weights file')
flags.DEFINE_string('image_path', '', 'path to image file')
flags.DEFINE_string('image_directory', './samples/', 'path to the directory of image files')
flags.DEFINE_string('image_type', 'png', 'file type of images')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_integer('cut_size', 100, 'resize images to')
flags.DEFINE_float('stride', 1/2, 'the stride of the sliding widow (this value is stride / cut_size)')
flags.DEFINE_float('margin', 1/16, 'the margin of image that will not be detected (this value is margin / cut_size)')
flags.DEFINE_integer('num_classes', 1, 'number of classes in the model')
flags.DEFINE_integer('batch_size', 12, 'batch size to detect')
flags.DEFINE_integer('dpi', 300, 'dpi of output image')
flags.DEFINE_string('output', './output/', 'output directory')
flags.DEFINE_boolean('output_image', True, 'whether output a result image marked with blue boxes')
flags.DEFINE_enum('output_type', 'boxes',
                  ['boxes', 'center', 'center_size', 'json', 'benchmark'],
                  'boxes: bounding boxes: (x_min y_min x_max y_max), '
                  'center: center coordinates: (x_center y_center), '
                  'center_size: center coordinates and size: (x_center y_center width*height), '
                  'json: .json file for future labeling in \"colabler\", '
                  'benchmark: for mAP calcuation: (\'particle\' confidence x_min y_min x_max y_max)')

def detect_tfrecord_batch(record_path, weights = './yolov3_tf2/yolov3_model.tf', classes = './yolov3_tf2/particle.names', num_classes = 1, size = 416, tiny = False, batch_size = 12):
    for filePath in glob.glob("./yolov3_tf2/tmp/*.txt"):
        os.remove(filePath)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    yolo = YoloV3(classes=num_classes)

    yolo.load_weights(weights).expect_partial()
    logging.info('weights loaded: ' + weights)

    class_names = [c.strip() for c in open(classes).readlines()]
    logging.info('classes loaded: ' + classes)

    dataset = load_tfrecord_dataset(record_path, classes, size)
    tot1 = 0
    tot2 = 0
    times = []
    imgs = []
    ret = []
    for img_raw, label_raw in iter(dataset):
        tot1 += 1
        imgs.append(img_raw)

        if tot1 % batch_size == 0:
            imgs = transform_images(imgs, size, random = False)
            t1 = time.time()
            boxes, scores, classes, nums = yolo.predict(imgs)
            t2 = time.time()
            times.append(t2 - t1)
            for k in range(len(nums)):
                ret.append([])
                for i in range(nums[k]):
                    ret[len(ret) - 1].append([boxes[k][i][0] * size, boxes[k][i][1] * size, boxes[k][i][2] * size, boxes[k][i][3] * size, scores[k][i]])
                tot2 += 1
            imgs = []
    if len(imgs) != 0:
        imgs = transform_images(imgs, size, random = False)
        boxes, scores, classes, nums = yolo.predict(imgs)
        for k in range(len(nums)):
            ret.append([])
            for i in range(nums[k]):
                ret[len(ret) - 1].append([boxes[k][i][0] * size, boxes[k][i][1] * size, boxes[k][i][2] * size, boxes[k][i][3] * size, scores[k][i]])
            tot2 += 1
    logging.info("total number of images: {}".format(tot1))
    logging.info('Time: {:.2f} ms'.format(np.average(np.asarray(times)) * 1000))
    return ret

def build_example(img_path):
    img_raw = open(img_path, 'rb').read()
    key = hashlib.sha256(img_raw).hexdigest()

    width = FLAGS.size
    height = FLAGS.size

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_path.encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_path.encode('utf8')])),
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

def Cut_Image(rawImage):
    xn, yn, zn = rawImage.shape
    imageSet = []
    imagePos = []
    for i in range(0, xn, int(FLAGS.cut_size * FLAGS.stride)):
        for j in range(0, yn, int(FLAGS.cut_size * FLAGS.stride)):
            if (i + FLAGS.cut_size < xn) and (j + FLAGS.cut_size < yn):
                imageSet.append(np.asarray(rawImage[i : i + FLAGS.cut_size, j : j + FLAGS.cut_size]))
                imagePos.append((i - int(FLAGS.cut_size * FLAGS.stride), j - int(FLAGS.cut_size * FLAGS.stride)))
    for i in range(len(imageSet)):
        pic = imageSet[i]
        plt.imshow(pic)
        plt.axis('off')
        fig = plt.gcf()
        fig.set_size_inches(FLAGS.size / FLAGS.dpi, FLAGS.size / FLAGS.dpi)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        fig.savefig('yolov3_tf2/tmp/' + str(i) + '.png', transparent = True, dpi = FLAGS.dpi, pad_inches = 0)
        plt.cla()
    
    return imageSet, imagePos

def yolo_detect(image_path):
    record_path = "yolov3_tf2/tmp.tfrecord"
    writer = tf.io.TFRecordWriter(record_path)
    for path in image_path:
        tf_example = build_example(path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    ret = detect_tfrecord_batch(record_path = record_path, weights = FLAGS.weights, classes = FLAGS.classes, num_classes = FLAGS.num_classes, size = FLAGS.size, batch_size = FLAGS.batch_size)
    os.remove(record_path)
    return ret

def Detect_Image(imagePos):    
    numberOfImage = len(imagePos)
    image_path = []
    for i in range(numberOfImage):
        image_path.append("yolov3_tf2/tmp/{}.png".format(i))
    boxRaw = yolo_detect(image_path)
    
    boxList = []
    for i in range(numberOfImage):
        px, py = imagePos[i]
        for box in boxRaw[i]:
            for i in range(4):
                if box[i] <= FLAGS.margin * FLAGS.size or box[i] >= (FLAGS.size - FLAGS.size * FLAGS.margin):
                    break
            else:
                tmp = np.asarray(box) / (FLAGS.size / FLAGS.cut_size)
                tmp[0] += py
                tmp[2] += py
                tmp[1] += px
                tmp[3] += px
                tmp[4] *= (FLAGS.size / FLAGS.cut_size)
                boxList.append(tmp)
    return boxList

def BoxMerge(box1, box2):
    return np.asarray([
            np.min([box1[0], box2[0]]), np.min([box1[1], box2[1]]),
            np.max([box1[2], box2[2]]), np.max([box1[3], box2[3]]), 
            np.max([box1[4], box2[4]])])

def DeleteCache():
    for filePath in glob.glob("yolov3_tf2/tmp/*.png"):
        os.remove(filePath)
    for filePath in glob.glob("yolov3_tf2/tmp/*.txt"):
        os.remove(filePath)

def TransposeBox(boxList, xn, yn):
    ansList = []
    for box in boxList:
        ansList.append([yn - box[2], xn - box[3], yn - box[0], xn - box[1], box[4]])
    return ansList

def Display_debug(image, boxLists = []):
    plt.figure(figsize=(20, 16))
    plt.imshow(image)
    ap = plt.gca()
    plt.axis('off')
    for boxList in boxLists:
        for po in boxList:
            rect = patches.Rectangle((po[0] - 0.5, po[1] - 0.5), po[2] - po[0], po[3] - po[1], linewidth = 1, edgecolor = 'b', facecolor = "none")
            ap.add_patch(rect)

def KNN_NMS(boxList):
    def take_conf(a):
        return a[4]
    def IOU(box1, box2):
        xx1 = np.max([box1[0], box2[0]])
        yy1 = np.max([box1[1], box2[1]])
        xx2 = np.min([box1[2], box2[2]])
        yy2 = np.min([box1[3], box2[3]])
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]) 
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]) 
        inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
        iou = inter_area / (area1 + area2 - inter_area + 1e-6)
        return iou
    def CreateTree(boxList):
        center = []
        for box in boxList:
            center.append([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        tree = KDTree(center)
        return tree, center
    
    boxList.sort(key = take_conf)
    boxList = boxList[::-1]
    tree, center = CreateTree(boxList)
    dis, ind = tree.query(x = center, k = 50, p = 2)
    tmpList = []
    inList = {}
    for i in range(len(dis)):
        for j in range(1, len(dis[i])):
            if ind[i][j] in inList:
                if IOU(boxList[i], boxList[ind[i][j]]) > 0.5:
                    break
        else:
            tmpList.append(boxList[i])
            inList[i] = 1
    return tmpList

def Add_Margin(img):
    x, y, z = np.asarray(img).shape
    tx = (int(x / FLAGS.cut_size) + 1) * FLAGS.cut_size + int(FLAGS.cut_size * FLAGS.stride) * 2
    ty = (int(y / FLAGS.cut_size) + 1) * FLAGS.cut_size + int(FLAGS.cut_size * FLAGS.stride) * 2
    img2 = Image.new("RGB", (ty, tx), (255, 255, 255))
    img2.paste(img, (int(FLAGS.cut_size * FLAGS.stride), int(FLAGS.cut_size * FLAGS.stride)))
    return img2

def WriteJson(name, allBox, xn, yn):
    with open(FLAGS.output + os.path.splitext(name)[0] + ".json", "w") as f:
        f.write("{")
        f.write("\"path\":\"C:\\\\Users\\\\Administrator\\\\Desktop\\\\dataset\\\\" + name + "\",")
        f.write("\"outputs\":{")
        f.write("\"object\":[")
        for i in range(len(allBox)):
            box = allBox[i]
            f.write("{\"name\":\"particle\",\"bndbox\":{")
            f.write("\"xmin\":" + str(int(box[0])) + ",")
            f.write("\"ymin\":" + str(int(box[1])) + ",")
            f.write("\"xmax\":" + str(int(box[2])) + ",")
            f.write("\"ymax\":" + str(int(box[3])) + "}}")
            if i != len(allBox) - 1:
                f.write(",")
        f.write("]")
        f.write("},")
        f.write("\"time_labeled\":1591181835610,")
        f.write("\"labeled\":true,")
        if FLAGS.image_type == 'tif':
            zn = 1
        else:
            zn = 3
        f.write("\"size\":{\"width\":" + str(yn) + ",\"height\":" + str(xn) + ",\"depth\":" + str(zn) + "}")
        f.write("}")
        
def WriteText(name, allBox, type):
    with open(FLAGS.output + os.path.splitext(name)[0] + ".txt", "w") as f:
        for i in range(len(allBox)):
            box = allBox[i]
            if type == 0:
                f.write(str(int(box[0])) + " ")
                f.write(str(int(box[1])) + " ")
                f.write(str(int(box[2])) + " ")
                f.write(str(int(box[3])))
                f.write('\n')
            if type == 1:
                f.write("particle ")
                f.write(str(float(box[4])) + " ")
                f.write(str(int(box[0])) + " ")
                f.write(str(int(box[1])) + " ")
                f.write(str(int(box[2])) + " ")
                f.write(str(int(box[3])))
                f.write('\n')
            if type == 2:
                f.write(str(int((box[0] + box[2]) / 2)) + " ")
                f.write(str(int((box[1] + box[3]) / 2)) + " ")
                f.write('\n')
            if type == 3:
                f.write(str(float((box[0] + box[2]) / 2)) + " ")
                f.write(str(float((box[1] + box[3]) / 2)) + " ")
                f.write(str(float((box[3] - box[1]) * (box[2] - box[0]))) + " ")
                f.write('\n')


def main(_argv):
    plt.rcParams['image.cmap'] = 'gray'
    if FLAGS.image_path != '':
        name = FLAGS.image_path
        DeleteCache()
        testImage = Add_Margin(Image.open(name).convert("RGB"))
        xn, yn, zn = np.asarray(testImage).shape
        imageSet, imagePos = Cut_Image(np.asarray(testImage))
        allBox = Detect_Image(imagePos)
        allBox = KNN_NMS(allBox)
        if FLAGS.output_image == True:
            Display_debug(Image.open(name).convert("L"), [allBox])
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.savefig(FLAGS.output + name[len(FLAGS.image_directory):][:-4] + ".png", dpi = FLAGS.dpi, bbox_inches = 'tight')
            plt.cla()
        if FLAGS.output_type == 'boxes':
            WriteText(os.path.basename(name), allBox, type = 0)
        elif FLAGS.output_type == 'benchmark':
            WriteText(os.path.basename(name), allBox, type = 1)
        elif FLAGS.output_type == 'center':
            WriteText(os.path.basename(name), allBox, type = 2)
        elif FLAGS.output_type == 'center_size':
            WriteText(os.path.basename(name), allBox, type = 3)
        elif FLAGS.output_type == 'json':
            WriteJson(os.path.basename(name), allBox, xn, yn)
    else:
        for name in glob.glob(FLAGS.image_directory + "*." + FLAGS.image_type):
            DeleteCache()
            testImage = Add_Margin(Image.open(name).convert("RGB"))
            xn, yn, zn = np.asarray(testImage).shape
            imageSet, imagePos = Cut_Image(np.asarray(testImage))
            allBox = Detect_Image(imagePos)
            allBox = KNN_NMS(allBox)
            if FLAGS.output_image == True:
                Display_debug(Image.open(name).convert("L"), [allBox])
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                plt.savefig(FLAGS.output + name[len(FLAGS.image_directory):][:-4] + ".png", dpi = FLAGS.dpi, bbox_inches = 'tight')
                plt.cla()
            if FLAGS.output_type == 'boxes':
                WriteText(os.path.basename(name), allBox, type = 0)
            elif FLAGS.output_type == 'benchmark':
                WriteText(os.path.basename(name), allBox, type = 1)
            elif FLAGS.output_type == 'center':
                WriteText(os.path.basename(name), allBox, type = 2)
            elif FLAGS.output_type == 'center_size':
                WriteText(os.path.basename(name), allBox, type = 3)
            elif FLAGS.output_type == 'json':
                WriteJson(os.path.basename(name), allBox, xn, yn)
    DeleteCache()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
