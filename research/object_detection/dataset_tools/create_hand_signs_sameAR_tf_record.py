# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert the Oxford pet dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/pets/

Example usage:
    python object_detection/dataset_tools/create_pet_tf_record.py \
        --data_dir=/home/user/pet \
        --output_dir=/home/user/pet/output
"""

import hashlib
import io
import logging
import os
import random
import re
import cv2
import json

import contextlib2
from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf

import sys

from copy import deepcopy

AMOUNT_OF_CUT = 54

sys.path.append(r"C:\Users\valentru\PycharmProjects\models\research\object_detection")


from dataset_tools import tf_record_creation_util
from utils import dataset_util
from utils import label_map_util

flags = tf.app.flags

flags.DEFINE_string('output_dir', 'Records_SameAR_27', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', '../data/SignLanguage_label_map.pbtxt',
                    'Path to label map proto')

flags.DEFINE_integer('num_shards', 1, 'Number of TFRecord shards')

FLAGS = flags.FLAGS


def get_label_from_folder_name(folder_name):
    label = folder_name.split('/')[-3]
    if label == 'N.0-right hand': return 'n0_right_hand'
    if label == 'N.0-left hand' : return 'n0_left_hand'
    if label == 'N.1-right hand': return 'n1_right_hand'
    if label == 'N.1-left hand':  return 'n1_left_hand'
    if label == 'N.2-right hand': return 'n2_right_hand'
    if label == 'N.2-left hand' : return 'n2_left_hand'
    if label == 'N.3-right hand': return 'n3_right_hand'
    if label == 'N.3-left hand':  return 'n3_left_hand'
    if label == 'N.4-right hand': return 'n4_right_hand'
    if label == 'N.4-left hand' : return 'n4_left_hand'
    if label == 'N.5-right hand': return 'n5_right_hand'
    if label == 'N.5-left hand':  return 'n5_left_hand'
    if label == 'N.6-right hand': return 'n6_right_hand'
    if label == 'N.6-left hand' : return 'n6_left_hand'
    if label == 'N.7-right hand': return 'n7_right_hand'
    if label == 'N.7-left hand':  return 'n7_left_hand'
    if label == 'N.8-right hand': return 'n8_right_hand'
    if label == 'N.8-left hand' : return 'n8_left_hand'
    if label == 'N.9-right hand': return 'n9_right_hand'
    if label == 'N.9-left hand':  return 'n9_left_hand'
    if label == 'N.10-right hand': return 'n10_right_hand'
    if label == 'N.10-left hand':  return 'n10_left_hand'
    if label == 'N.HELP-right hand': return 'HELP_right_hand'
    if label == 'N.HELP-left hand':  return 'HELP_left_hand'
    if label == 'N.NO-right hand': return 'NO_right_hand'
    if label == 'N.NO-left hand':  return 'NO_left_hand'
    if label == 'N.YES-right hand': return 'YES_right_hand'
    if label == 'N.YES-left hand':  return 'YES_left_hand'

    print('did not match folder naming convention')
    exit(0)
    return label


def create_tf_example(content, label_map_dict):
    img_path = content[0]
    json_path = content[1]





    image = cv2.imread(img_path)
    width = image.shape[1]
    height = image.shape[0]

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []



    with open(json_path) as f:

        xs = [float(data["Annotations"][0]['PointTopLeft'].split(',')[0]),float(data["Annotations"][0]['PointTopRight'].split(',')[0]),\
              float(data["Annotations"][0]['PointBottomLeft'].split(',')[0]),float(data["Annotations"][0]['PointBottomRight'].split(',')[0])]
        ys = [float(data["Annotations"][0]['PointTopLeft'].split(',')[1]),float(data["Annotations"][0]['PointTopRight'].split(',')[1]),\
              float(data["Annotations"][0]['PointBottomLeft'].split(',')[1]),float(data["Annotations"][0]['PointBottomRight'].split(',')[1])]

        data = json.load(f)
        xmin = min(xs)
        ymin = min(ys)
        xmax = max(xs)
        ymax = max(ys)

        if "NewVideos" in json_path:
            class_name = data["Annotations"][0]["Label"]
            if class_name[0] == ' ':
                class_name = class_name[1:]
            if class_name[-1] == ' ':
                class_name = class_name[:-1]

            class_name = class_name.replace(' ', '_')
            class_name = class_name.replace('__', '_')
        else:
            class_name = get_label_from_folder_name(json_path)



        # bring to same ratio
        if(image.shape[0] == 480 and image.shape[1] == 640):
            cut_x_left = min(AMOUNT_OF_CUT / 2, int(xmin))
            cut_x_right = min(AMOUNT_OF_CUT / 2 + (AMOUNT_OF_CUT / 2 - cut_x_left), int(image.shape[1] - int(xmax)))

            if cut_x_left + cut_x_right < AMOUNT_OF_CUT:
                cut_x_left = min(AMOUNT_OF_CUT / 2 + max(0, AMOUNT_OF_CUT / 2 - cut_x_right), int(xmin))

            imageTruncated = deepcopy(image[:,int(cut_x_left):image.shape[1] - int(cut_x_right)])

            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 0), 3)
            cv2.putText(image, class_name,
                        (int(xmin) + 2, int(ymin) + 12),
                        0, 8 * 1.2e-4 * image.shape[0],
                        (0, 255, 0), 2)
            cv2.imshow("original_image", image)

            xmin = xmin - cut_x_left
            xmax = xmax - cut_x_right
            width = imageTruncated.shape[1]
            image_to_draw = deepcopy(imageTruncated)
            cv2.rectangle(image_to_draw, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 0), 3)
            cv2.putText(image_to_draw, class_name,
                        (int(xmin) + 2, int(ymin) + 12),
                        0, 8 * 1.2e-4 * image_to_draw.shape[0],
                        (0, 255, 0), 2)
            cv2.imshow("truncated_image", imageTruncated)
            cv2.waitKey(0)
            print(image.shape)
            print(imageTruncated.shape)

            cv2.imwrite("truncated_pic.jpg",imageTruncated)

            with tf.gfile.GFile("truncated_pic.jpg", 'rb') as fid:
                encoded_jpg = fid.read()

            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = PIL.Image.open(encoded_jpg_io)
            if image.format != 'JPEG':
                raise ValueError('Image format not JPEG')
            key = hashlib.sha256(encoded_jpg).hexdigest()
        else:
            with tf.gfile.GFile(img_path, 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = PIL.Image.open(encoded_jpg_io)
            if image.format != 'JPEG':
                raise ValueError('Image format not JPEG')
            key = hashlib.sha256(encoded_jpg).hexdigest()

        xmins.append(xmin / width)
        ymins.append(ymin / height)
        xmaxs.append(xmax / width)
        ymaxs.append(ymax / height)


        # cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 0), 3)
        # cv2.putText(image, class_name,
        #             (int(xmin) + 2, int(ymin) + 12),
        #             0, 8 * 1.2e-4 * image.shape[0],
        #             (0, 255, 0), 2)
        # cv2.imshow("image", image)
        # cv2.waitKey(0)



    classes_text.append(class_name.encode('utf8'))
    classes.append(label_map_dict[class_name])


    feature_dict = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          img_path.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          img_path.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes)
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example

def create_tf_example_sequence(content, label_map_dict, index, name="train"):
    img_path = content[0]
    json_path = content[1]

    path_for_debug = "Data_Annotated"


    image = cv2.imread(img_path)
    width = image.shape[1]
    height = image.shape[0]

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []



    with open(json_path) as f:

        data = json.load(f)

        xs = [float(data["Annotations"][0]['PointTopLeft'].split(',')[0]),
              float(data["Annotations"][0]['PointTopRight'].split(',')[0]), \
              float(data["Annotations"][0]['PointBottomLeft'].split(',')[0]),
              float(data["Annotations"][0]['PointBottomRight'].split(',')[0])]
        ys = [float(data["Annotations"][0]['PointTopLeft'].split(',')[1]),
              float(data["Annotations"][0]['PointTopRight'].split(',')[1]), \
              float(data["Annotations"][0]['PointBottomLeft'].split(',')[1]),
              float(data["Annotations"][0]['PointBottomRight'].split(',')[1])]


        xmin = min(xs)
        ymin = min(ys)
        xmax = max(xs)
        ymax = max(ys)


        class_name = data["Annotations"][0]["Label"]
        class_name = class_name.replace(" ","_")
        # bring to same ratio
        if(image.shape[0] == 480 and image.shape[1] == 640):
            cut_x_left = min(AMOUNT_OF_CUT / 2, int(xmin))
            cut_x_right = min(AMOUNT_OF_CUT / 2 + (AMOUNT_OF_CUT / 2 - cut_x_left), int(image.shape[1] - int(xmax)))
            print("I am here, in Too Big Image")

            if cut_x_left + cut_x_right < AMOUNT_OF_CUT:
                cut_x_left = min(AMOUNT_OF_CUT / 2 + max(0, AMOUNT_OF_CUT / 2 - cut_x_right), int(xmin))

            imageTruncated = deepcopy(image[:,int(cut_x_left):image.shape[1] - int(cut_x_right)])

            # cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 0), 3)
            # cv2.putText(image, class_name,
            #             (int(xmin) + 2, int(ymin) + 12),
            #             0, 8 * 1.2e-4 * image.shape[0],
            #             (0, 255, 0), 2)
            # cv2.imshow("original_image", image)

            xmin = xmin - cut_x_left
            xmax = xmax - cut_x_right
            width = imageTruncated.shape[1]

            # code for debug
            image_to_draw = deepcopy(imageTruncated)
            # cv2.rectangle(image_to_draw, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 0), 3)
            # cv2.putText(image_to_draw, class_name,
            #             (int(xmin) + 2, int(ymin) + 12),
            #             0, 8 * 1.2e-4 * image_to_draw.shape[0],
            #             (0, 255, 0), 2)
            # cv2.imshow("truncated_image", image_to_draw)
            # cv2.waitKey(0)
            # print(image.shape)
            # print(imageTruncated.shape)
            # # cv2.imwrite(os.path.join(path_for_debug,name + str(index) + ".jpg"), image_to_draw)

            # write image for reading afterwards and encoding it
            cv2.imwrite("truncated_pic.jpg", imageTruncated)
            with tf.gfile.GFile("truncated_pic.jpg", 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = PIL.Image.open(encoded_jpg_io)
            if image.format != 'JPEG':
                raise ValueError('Image format not JPEG')
            key = hashlib.sha256(encoded_jpg).hexdigest()

        else:
            # debug purposes
            image_to_draw = deepcopy(image)
            cv2.rectangle(image_to_draw, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 0), 3)
            cv2.putText(image_to_draw, class_name,
                        (int(xmin) + 2, int(ymin) + 12),
                        0, 8 * 1.2e-4 * image_to_draw.shape[0],
                        (0, 255, 0), 2)
            # cv2.imwrite(os.path.join(path_for_debug, name + str(index) + ".jpg"), image_to_draw)
            # cv2.imshow("image", image_to_draw)
            # cv2.waitKey(0)
            print("I am here, in Normal Big Image")


            # encode image
            with tf.gfile.GFile(img_path, 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = PIL.Image.open(encoded_jpg_io)
            if image.format != 'JPEG':
                raise ValueError('Image format not JPEG')
            key = hashlib.sha256(encoded_jpg).hexdigest()

        xmins.append(xmin / width)
        ymins.append(ymin / height)
        xmaxs.append(xmax / width)
        ymaxs.append(ymax / height)


        # cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 0), 3)
        # cv2.putText(image, class_name,
        #             (int(xmin) + 2, int(ymin) + 12),
        #             0, 8 * 1.2e-4 * image.shape[0],
        #             (0, 255, 0), 2)
        # cv2.imshow("image", image)
        # cv2.waitKey(0)



    classes_text.append(class_name.encode('utf8'))
    classes.append(label_map_dict[class_name])


    feature_dict = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          img_path.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          img_path.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes)
    }
    print("I am here, in the end")

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example

def create_tf_example_merged_labels(content, label_map_dict):
    img_path = content[0]
    json_path = content[1]

    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    imagecv = cv2.imread(img_path)
    width = imagecv.shape[1]
    height = imagecv.shape[0]

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []
    image = cv2.imread(img_path)
    with open(json_path) as f:
        data = json.load(f)
        xmin = float(data["Annotations"][0]['PointTopLeft'].split(',')[0])
        ymin = float(data["Annotations"][0]['PointTopLeft'].split(',')[1])
        xmax = float(data["Annotations"][0]['PointBottomRight'].split(',')[0])
        ymax = float(data["Annotations"][0]['PointBottomRight'].split(',')[1])

        print("I am here, in JSON")

        xmins.append(xmin / width)
        ymins.append(ymin / height)
        xmaxs.append(xmax / width)
        ymaxs.append(ymax / height)

        class_name = data["Annotations"][0]["Label"]
        if "NewVideos" in json_path:
            print("I am here, in NewVideos")
            class_name = class_name.replace(' ', '')
            class_name = class_name.lower()
            class_name = (class_name.replace("left", "")).replace("right", "")
        else:

            class_name = json_path.split('/')[-3]
            class_name = ((class_name.replace(".", "")).replace("-", "")).replace(" ", "")
            class_name = class_name.lower()
            class_name = (class_name.replace("left", "")).replace("right", "")
            if "yes" in class_name or "no" in class_name or "help" in class_name:
                class_name = class_name[1:]
            print("I am here, in OldVideos")
        print(class_name)

        # cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 0), 3)
        # cv2.putText(image, class_name,
        #             (int(xmin) + 2, int(ymin) + 12),
        #             0, 8 * 1.2e-4 * image.shape[0],
        #             (0, 255, 0), 2)
        # cv2.imshow("image", image)
        # cv2.waitKey(1)



    classes_text.append(class_name.encode('utf8'))
    classes.append(label_map_dict[class_name])


    feature_dict = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          img_path.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          img_path.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes)
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def create_tf_record(output_filename,
                     num_shards,
                     label_map_dict,
                     examples, name):

  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_filename, num_shards)
    for idx, example in enumerate(examples):
        print(idx)
        tf_example = create_tf_example_sequence(example, label_map_dict, idx, name)
        if tf_example:
          shard_idx = idx % num_shards
          output_tfrecords[shard_idx].write(tf_example.SerializeToString())



def convert_data_from_Andi(file_of_paths):
    with open(file_of_paths) as f:
        content = f.readlines()

    content = [x.strip() for x in content]

    newContent = []
    startIndex = len("/home/dyve/Desktop/BigData/Andi/DyVe/Sign_language-Full_Annotation/")
    antet = "D:\Hand Detection\HandSigns"

    for con in content:
        halfs = con.split(" gt: ")
        image_path = antet + "/" + halfs[0][startIndex:]
        json_path = antet + "/" + halfs[1][startIndex:]

        newContent.append([image_path, json_path])
    return newContent

def convert_data(file_of_paths):
    with open(file_of_paths) as f:
        content = f.readlines()

    content = [x.strip() for x in content]

    newContent = []

    for con in content:
        halfs = con.split(" nnn ")
        image_path = halfs[0]
        json_path = halfs[1]

        newContent.append([image_path, json_path])
    return newContent

def main(_):
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
  print(label_map_dict)

  file_of_paths_train = r"D:\Hand Detection\HandSigns\DatasetHandsSequence\27_classes_train.txt"
  file_of_paths_val = r"D:\Hand Detection\HandSigns\DatasetHandsSequence\27_classes_test.txt"

  content_train = convert_data(file_of_paths_train)
  content_val = convert_data(file_of_paths_val)


  train_output_path = os.path.join(FLAGS.output_dir, 'hand_signs_train_sameAR_27.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'hand_signs_val_sameAR_27.record')

  create_tf_record(
      train_output_path,
      FLAGS.num_shards,
      label_map_dict,
      content_train,
      name="train")

  create_tf_record(
      val_output_path,
      FLAGS.num_shards,
      label_map_dict,
      content_val,
      name="test")



if __name__ == '__main__':
  tf.app.run()
