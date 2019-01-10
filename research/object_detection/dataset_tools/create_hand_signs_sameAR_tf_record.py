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
AMOUNT_OF_PADDING = 31

sys.path.append("/home/dyve/PycharmProjects/models/research/object_detection")


from dataset_tools import tf_record_creation_util
from utils import dataset_util
from utils import label_map_util
import imgaug as ia
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

flags = tf.app.flags

flags.DEFINE_string('output_dir', 'Records_SameAR_27', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', '../data/SignLanguage_label_map.pbtxt',
                    'Path to label map proto')

flags.DEFINE_integer('num_shards', 1, 'Number of TFRecord shards')

FLAGS = flags.FLAGS

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        # iaa.Fliplr(0.5), # horizontally flip 50% of all images
        # iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        # sometimes(iaa.CropAndPad(
        #     percent=(-0.05, 0.1),
        #     pad_mode=ia.ALL,
        #     pad_cval=(0, 255)
        # )),
        sometimes(iaa.Affine(
            scale={"x": (0.7, 1.2), "y": (0.7, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.1, 0.2), "y": (-0.1, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-5, 5), # rotate by -45 to +45 degrees
            # shear=(-16, 16), # shear by -16 to +16 degrees
            # order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            # cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((2, 5),
            [
                # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 0.1)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(3, 3)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 3)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.5, 1.0)), # sharpen images
                # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                # iaa.SimplexNoiseAlpha(iaa.OneOf([
                #     iaa.EdgeDetect(alpha=(0.5, 1.0)),
                #     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                # ])),
                # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255)), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.03)), # randomly remove up to 10% of the pixels
                    # iaa.CoarseDropout((0.02, 0.05), size_percent=(0.01, 0.03)),
                ]),
                # iaa.Invert(0.05), # invert color channels
                iaa.Add((-10, 10)), # change brightness of images (by -10 to 10 of original value)
                # iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                # iaa.OneOf([
                #     iaa.Multiply((0.5, 1.5)),
                    # iaa.FrequencyNoiseAlpha(
                    #     exponent=(-4, 0),
                    #     first=iaa.Multiply((0.5, 1.5)),
                    #     second=iaa.ContrastNormalization((0.5, 2.0))
                    # )
                # ]),
                # iaa.ContrastNormalization((0.5, 2.0)), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 0.5)),
                # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                # sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
        )
    ],
    random_order=True
)


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


listOfUnwantedLabels = ['HELP_right_hand','HELP_left_hand','NO_right_hand','NO_left_hand','YES_right_hand','YES_left_hand']
nrOfClasses = []
badImagesIndex = []




def create_tf_example_sequence_ResizeUp(content, label_map_dict, index, name="train"):
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
        # if class_name in listOfUnwantedLabels:
        #     print("Unwanted label")
        #     return None

        nrOfClasses.append(class_name)
        if class_name == 'none':
            print("Class None")
        # bring to same ratio
        if(image.shape[0] == 288 and image.shape[1] == 352):
            # cut_x_left = min(AMOUNT_OF_CUT / 2, int(xmin))
            # cut_x_right = min(AMOUNT_OF_CUT / 2 + (AMOUNT_OF_CUT / 2 - cut_x_left), int(image.shape[1] - int(xmax)))
            print("I am here, in Too Big Image")

            # if cut_x_left + cut_x_right < AMOUNT_OF_CUT:
                # cut_x_left = min(AMOUNT_OF_CUT / 2 + max(0, AMOUNT_OF_CUT / 2 - cut_x_right), int(xmin))

            # imageTruncated = deepcopy(image[:,int(cut_x_left):image.shape[1] - int(cut_x_right)])

            img1 = deepcopy(image)
            # replicate = cv2.copyMakeBorder(img1, 0, 0, 15, 16, cv2.BORDER_REPLICATE)
            # reflect = cv2.copyMakeBorder(img1, 0, 0, 15, 16, cv2.BORDER_REFLECT)
            reflect101 = cv2.copyMakeBorder(img1, 0, 0, 15, 16, cv2.BORDER_REFLECT_101)
            # wrap = cv2.copyMakeBorder(img1, 0, 0, 15, 16, cv2.BORDER_WRAP)
            # constant = cv2.copyMakeBorder(img1, 0, 0, 15, 16, cv2.BORDER_CONSTANT, value=[255,255,255])

            # cv2.imshow("original_image", image)
            # cv2.imshow("replicate", replicate)
            # cv2.imshow("reflect", reflect)
            # cv2.imshow("reflect101", reflect101)
            # cv2.imshow("wrap", wrap)
            # cv2.imshow("constant", constant)
            # cv2.waitKey(0)

            # reflect101 = cv2.resize(reflect101,(640,480),interpolation=cv2.INTER_LANCZOS4)
            # cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 0), 3)
            # cv2.putText(image, class_name,
            #             (int(xmin) + 2, int(ymin) + 12),
            #             0, 8 * 1.2e-4 * image.shape[0],
            #             (0, 255, 0), 2)
            # cv2.imshow("original_image", image)

            xmin = xmin + 15
            xmax = xmax + 15
            width = image.shape[1]

            # code for debug
            image_to_draw = deepcopy(reflect101)

            print(image.shape)
            print(reflect101.shape)
            # # cv2.imwrite(os.path.join(path_for_debug,name + str(index) + ".jpg"), image_to_draw)

            # write image for reading afterwards and encoding it
            cv2.imwrite("truncated_pic.jpg", image_to_draw)
            with tf.gfile.GFile("truncated_pic.jpg", 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            if len(encoded_jpg)<100:
                badImagesIndex.append(1)
                return False
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
        if class_name in listOfUnwantedLabels:
            print("Unwanted label")
            return None

        nrOfClasses.append(class_name)
        if class_name == 'none':
            print("Class None")
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
            cv2.imwrite("truncated_pic.jpg", image_to_draw)
            with tf.gfile.GFile("truncated_pic.jpg", 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            if len(encoded_jpg)<100:
                badImagesIndex.append(1)
                return False
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



def create_tf_record(output_filename,
                     num_shards,
                     label_map_dict,
                     examples, name):

  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_filename, num_shards)
    index= 0
    idxValue = 0
    for idx, example in enumerate(examples):
        print(idx)
        tf_example = create_tf_example_sequence_ResizeUp(example, label_map_dict, idx, name)
        idxValue = idx
        if tf_example:
          shard_idx = idx % num_shards
          output_tfrecords[shard_idx].write(tf_example.SerializeToString())
        else:
            index = index+1
  nrOfImages = idxValue - index
  print("Number of images = " ,str(nrOfImages))
  print("BAD IMAGES + ", str(len(badImagesIndex)))




def convert_data(file_of_paths):
    with open(file_of_paths) as f:
        content = f.readlines()

    content = [x.strip() for x in content]

    newContent = []

    for con in content:
        halfs = con.split(" NNN ")
        image_path = halfs[0]
        json_path = halfs[1]

        newContent.append([image_path, json_path])
    return newContent

def main(_):
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
  print(label_map_dict)


  # file_of_paths_train = "/media/dyve/BigData/HandDetection/Sequence_Dataset/newTrainVal/signLanguage_train.txt"
  file_of_paths_val = "/media/dyve/BigData/HandDetection/Sequence_Dataset/newTrainVal/signLanguage_val.txt"

  # content_train = convert_data(file_of_paths_train)
  content_val = convert_data(file_of_paths_val)


  # train_output_path = os.path.join(FLAGS.output_dir, 'hand_signs_train_sameAR_27_newTrain.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'hand_signs_val_sameAR_27_newVal.record')

  # create_tf_record(
  #     train_output_path,
  #     FLAGS.num_shards,
  #     label_map_dict,
  #     content_train,
  #     name="train")
  #
  create_tf_record(
      val_output_path,
      FLAGS.num_shards,
      label_map_dict,
      content_val,
      name="test")

  elementsPerClass = []
  labels = []
  for key in label_map_dict:
      labels.append(key)
      index = 0
      for c in nrOfClasses:
          if key == c:
              index = index + 1

      elementsPerClass.append(index)

  objects = tuple(labels)
  y_pos = np.arange(len(objects))

  plt.bar(y_pos, elementsPerClass, align='center', alpha=0.5)
  plt.xticks(y_pos, objects)
  plt.ylabel('Nr of examples')
  plt.title('Data Distribution')

  plt.show()

  print(elementsPerClass)

if __name__ == '__main__':
  tf.app.run()
