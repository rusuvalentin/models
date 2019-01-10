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

import contextlib2
from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf
from glob import glob
import json
import matplotlib.pyplot as plt
from random import shuffle
import sys
sys.path.append("/home/dyve/PycharmProjects/models/research/object_detection")

from dataset_tools import tf_record_creation_util
from utils import dataset_util
from utils import label_map_util


DEBUG_DISTRIBUTION = True
CROP_SIZE = 0 #pixels to crop xmin and xmax
AMOUNT_OF_CUT = 54


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir',
                    '/media/dyve/BigData/HandDetection/Sequence_Dataset/',
                    'Data directory to raw SignLanguage dataset.')

flags.DEFINE_string('root_dir',
                    r'/home/dyve/PycharmProjects/models/research/object_detection/',
                    'Root directory to raw SignLanguage dataset.')
flags.DEFINE_string('image_visualization_dir',
                    FLAGS.root_dir + 'Sequence_Output',
                    'Data directory to raw SignLanguage dataset.')
flags.DEFINE_string('output_dir', os.path.join(FLAGS.root_dir, 'dataset_tools/Records_SameAR_27'), 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', os.path.join(FLAGS.root_dir,'data/SignLanguage_label_map.pbtxt'),
                    'Path to label map proto')

flags.DEFINE_integer('num_shards', 1, 'Number of TFRecord shards')


global num_images_in_train
global num_images_in_test
global num_images_ignored
global save_train_filelist
global save_test_filelist

def get_class_name_from_filename(file_name):
  """Gets the class name from a file.

  Args:
    file_name: The file name to get the class name from.
               ie. "american_pit_bull_terrier_105.jpg"

  Returns:
    A string of the class name.
  """
  match = re.match(r'([A-Za-z_]+)(_[0-9]+\.jpg)', file_name, re.I)
  return match.groups()[0]

all_class_names = []
all_sizes = []



def encode_image(image_path):
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    return encoded_jpg, key

#sameAR = 0 -> 640 x 480 images are truncated
#sameAR = 1 -> 352 x 288 images are padded
def dict_to_tf_example(data,
                       label_map_dict,augment=False, sameAR = 1):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    mask_path: String path to PNG encoded mask.
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset directory holding the actual image data.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    faces_only: If True, generates bounding boxes for pet faces.  Otherwise
      generates bounding boxes (as well as segmentations for full pet bodies).
    mask_type: 'numerical' or 'png'. 'png' is recommended because it leads to
      smaller file sizes.

    Returns:
    example: The converted tf.Example.

    Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    img_path = data[0]
    parsed_results = data[2]
    json_path = data[1]
    if(json_path=='None'):
        raise ValueError('Image had no JSON')
    if (parsed_results[0]==True):
        raise ValueError('Image isBadImage')

    image = cv2.imread(img_path)
    width = image.shape[1]
    height = image.shape[0]

    if ([width,height] in all_sizes)==False:
        all_sizes.append([width, height])


    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []

    xmin = max(float(parsed_results[4].split(',')[0]),0)
    xmax = min(float(parsed_results[5].split(',')[0]),width)
    ymin = max(float(parsed_results[4].split(',')[1]),0)
    ymax = min(float(parsed_results[7].split(',')[1]),height)

    #print(parsed_results)
    if (xmin>=xmax or xmin<0 or xmax>width) :
        #print(img_path,' ',parsed_results)
        #print(xmin,xmax,ymin,ymax)
        #test_visualization(img_path, parsed_results)
        raise ValueError('Image isBadImage')

    if(ymin>=ymax or ymin<0 or ymax>height):
        #test_visualization(img_path, parsed_results)
        raise ValueError('Image isBadImage')

    path_to_read_from = img_path


    if sameAR == 0:
        if (height==480 and width==640):
            #print('Bring to same Aspect Ratio')
            cut_x_left = min(AMOUNT_OF_CUT / 2, int(xmin))
            cut_x_right = min(AMOUNT_OF_CUT / 2 + (AMOUNT_OF_CUT / 2 - cut_x_left), int(image.shape[1] - int(xmax)))

            if cut_x_left + cut_x_right < AMOUNT_OF_CUT:
                cut_x_left = min(AMOUNT_OF_CUT / 2 + max(0, AMOUNT_OF_CUT / 2 - cut_x_right), int(xmin))

            imageTruncated = image[:, int(cut_x_left):image.shape[1] - int(cut_x_right)]

            xmin = xmin - cut_x_left
            xmax = xmax - cut_x_left
            width = imageTruncated.shape[1]
            height = imageTruncated.shape[0]

            cv2.imwrite("truncated_pic.jpg", imageTruncated)
            encoded_jpg, key = encode_image("truncated_pic.jpg")
        else:
            # encode image
            encoded_jpg, key = encode_image(path_to_read_from)
    else:
        if (height == 288 and width == 352):
            left_pad_amount = 15
            right_pad_amount = 16

            paddedImage = cv2.copyMakeBorder(image, top=0, bottom=0, left=left_pad_amount, right=right_pad_amount,
                                        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

            xmin = xmin + left_pad_amount
            xmax = xmax + left_pad_amount
            width = paddedImage.shape[1]
            height = paddedImage.shape[0]
            # print(paddedImage.shape)
            # cv2.imshow("paddedImage",paddedImage)
            # cv2.waitKey(0)

            cv2.imwrite("padded_pic.jpg", paddedImage)
            encoded_jpg, key = encode_image('padded_pic.jpg')
        else:
            # cv2.imshow("original", image)
            # cv2.waitKey(0)
            image = cv2.resize(image,(383, 288))
            cv2.imwrite("downsized_pic.jpg", image)
            encoded_jpg, key = encode_image("downsized_pic.jpg")


    xmins.append(xmin / width)
    ymins.append(ymin / height)
    xmaxs.append(xmax / width)
    ymaxs.append(ymax / height)
    class_name = parsed_results[-1]

    classes_text.append(class_name.encode('utf8'))
    classes.append(label_map_dict[class_name])

    if ((class_name in all_class_names)==False):
        all_class_names.append(class_name)

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
                     examples,isTrain=False):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    num_shards: Number of shards for output file.
    label_map_dict: The label map dictionary.
    annotations_dir: Directory where annotation files are stored.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
    faces_only: If True, generates bounding boxes for pet faces.  Otherwise
      generates bounding boxes (as well as segmentations for full pet bodies).
    mask_type: 'numerical' or 'png'. 'png' is recommended because it leads to
      smaller file sizes.
  """
  global num_images_in_train
  global num_images_in_test
  global num_images_ignored
  global save_train_filelist
  global save_test_filelist


  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_filename, num_shards)

    total_results = examples[:]
    for idx, data in enumerate(examples):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(examples))

      try:
        tf_example = dict_to_tf_example(
            data,
            label_map_dict,False,0)
        if tf_example:
          shard_idx = idx % num_shards
          output_tfrecords[shard_idx].write(tf_example.SerializeToString())
          if isTrain:
              num_images_in_train+=1
              with open(save_train_filelist,'a') as f:
                  f.write(data[0]+' gt: '+data[1]+'\n')
          else:
              num_images_in_test+=1
              with open(save_test_filelist,'a') as f:
                  f.write(data[0]+' gt: '+data[1]+'\n')

          #augment some classes
          # if (('none' in data[2][-1]==False) and isTrain):
          #     total_results.append(data)
          #     tf_example = dict_to_tf_example(
          #         data,
          #         label_map_dict,augment=True)
          #     shard_idx = idx % num_shards
          #     output_tfrecords[shard_idx].write(tf_example.SerializeToString())
          #
          #     num_images_in_train += 1
          #     with open(save_train_filelist, 'a') as f:
          #         f.write(data[0] + ' gt: ' + data[1] + '\n')


      except ValueError:
        num_images_ignored +=1
        #logging.warning('Invalid example: ignoring.')

    bad_img_no = 0
    class_dict = {}

    for idx, result in enumerate(total_results):
        if (result[2][0] == True):
            bad_img_no += 1
        else:
            key = result[2][-1]
            if key in class_dict:
                class_dict[key] += 1
            else:
                class_dict[key] = 1

    print('Number of bad images : ', bad_img_no, ' => ', len(total_results) - bad_img_no, ' usable object images')
    for class_label, numbers in class_dict.items():
        print(class_label, ' ', numbers)
    print('***' * 10 + str(len(class_dict)))

    if (DEBUG_DISTRIBUTION):
        class_dict_n = list(range(len(class_dict.keys())))
        plt.bar(class_dict_n, class_dict.values(), color='g')
        plt.xticks(class_dict_n, [key.replace('_hand', '') for key in class_dict.keys()])
        plt.xticks(rotation=90)

        plt.title('Distribution of SignLanguage Classes')
        plt.xlabel('Classes')
        plt.ylabel('Examples')
        plt.show()


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

def format_annotation_data(label_from_folder_name,json_path,annotation_data):
    if (label_from_folder_name):
        label_name = get_label_from_folder_name(json_path)
        return label_name

    if annotation_data[0]==' ':
        annotation_data = annotation_data[1:]
    if annotation_data[-1]==' ':
        annotation_data = annotation_data[:-1]

    annotation_data = annotation_data.replace(' ','_')
    annotation_data = annotation_data.replace('__', '_')

    return annotation_data

def parse_json(json_path,label_from_folder_name=False, original=False):

    if (original==True):
        json_path=json_path.replace('ForTrainingAnnotations','ForTrainingAnnotations_original')
        if (os.path.exists(json_path)==False):
            return [True, 0, 0, 0, 0, 0, 0, 0, 0]

    with open(json_path) as f:
        try:
            data=json.load(f)
        except:
            return [True, 0, 0, 0, 0, 0, 0, 0, 0]


    annotation_data = data['Annotations']
    #print(annotation_data)
    if len(annotation_data)==0:
        #print(json_path,' has no Annotations')
        return [True,0,0,0,0,0,0,0,0]

    annotation_data = annotation_data[0]
    if data['IsBadImage']==True :
        print(json_path,' isBadImage')
        exit(0)
    if annotation_data['RealAngle']!=0.0:
        print(json_path,' has different RealAngle')
        #exit(0)
    if annotation_data['Angle']!=0.0:
        print(json_path,' has different Angle')
        #exit(0)
    if str(annotation_data['Type'])!='manual':
        print(json_path,' has different Type than manual')
        exit(0)

    annotation_data['Label'] = format_annotation_data(label_from_folder_name,json_path,annotation_data['Label'])

    return [data['IsBadImage'],annotation_data['RealAngle'],annotation_data['Angle'],str(annotation_data['Type']),
            str(annotation_data['PointTopLeft']),str(annotation_data['PointTopRight']),
            str(annotation_data['PointBottomLeft']),str(annotation_data['PointBottomRight']),str(annotation_data['Label'])]

global image_no
global filelist_test
# def test_visualization(image_path,parsed_results):
#     global image_no
#     global filelist_test
#
#
#
#     with open(filelist_test,'a') as f:
#         f.write(image_path+'\n')
#
#     print('-+-+-'*10)
#     print(image_path)
#     print(parsed_results)
#
#     image = cv2.imread(image_path)
#     xmin = int(float(parsed_results[4].split(',')[0]))
#     xmax = int(float(parsed_results[5].split(',')[0]))
#     ymin = int(float(parsed_results[4].split(',')[1]))
#     ymax = int(float(parsed_results[7].split(',')[1]))
#
#     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
#     cv2.imwrite(FLAGS.image_visualization_dir + '/' + str(image_no)+'.jpg',image)
#     image_no+=1

def get_data_from_folders(data_dir_list, is_train=True):

    total_results = []
    number_of_NOs_added = 0

    for folder in data_dir_list:
        none_class_results = []
        NO_class_results = []
        #folder= folder.replace(' ','\ ')
        print(folder)
        print(folder.split('/')[-1] + ' ' + '---' * 10)

        full_image_list = sorted(glob(folder + '/*.jpg'))
        print('Full image list has size : ',len(full_image_list))

        label_from_folder_name = False
        json_list = glob(folder + '/ForTrainingAnnotations/*.json')

        print('Json list has size : ', len(json_list))
        shuffle(full_image_list)
        image_gt_list = []
        for img_file in full_image_list:
            img_key = img_file.split('\\')[-1][:-4].split('/')[-1]
            img_key=img_key+"_forTraining.json"
            for json_file in json_list:
                if (img_key in json_file.split('/')[-1]):
                    image_gt_list.append([img_file,json_file])
                    break

        print('Paired images and jsons : ',len(image_gt_list))

        consecutive_nones=0
        consecutive_none_frames = []
        consecutive_NO_frames = []

        for image_gt_pair in image_gt_list:
            image_path = image_gt_pair[0]
            json_path = image_gt_pair[1]

            add_result_to_filelist = False
            result = parse_json(json_path,label_from_folder_name)

            if (result[-1]=='none'):
                consecutive_nones +=1
                consecutive_none_frames.append([image_path,json_path,result])
            else:
                if (len(consecutive_none_frames)>0):
                    none_class_results.extend(consecutive_none_frames[int(consecutive_nones / 3):int(-consecutive_nones / 3)])
                    consecutive_none_frames = []
                    consecutive_nones=0

            #if (result[0]!=True and 'NO' in result[-1]):
            #    result_original = parse_json(json_path,label_from_folder_name,original=True)
            #    if (result_original[0]!=True and ('SEQUENCE' in folder)):
            #        if(result_original[-1]==result[-1]):
            #            number_of_NOs_added+=1
            #            total_results.append([image_path, json_path, result])
            #            test_visualization(image_path, result)

            #    if (result_original[0]==True and ('CLASSIFIER'in folder)):
            #        consecutive_NO_frames.append([image_path,json_path,result])
            #else:
            #    consecutive_NOs = len(consecutive_NO_frames)
            #    if (consecutive_NOs>0):
            #        print('======='*10+' '+str(len(NO_class_results)))
            #        NO_class_results.extend(consecutive_NO_frames[int(consecutive_NOs/2):])
            #        for element in consecutive_NO_frames[int(consecutive_NOs/2):]:
            #            test_visualization(element[0], element[2])
            #        consecutive_NO_frames = []

            json_list.append(json_path)

            #if (result[0]!=True and result[-1]!='none' and ('NO' in result[-1])==False and
            #    ('n10' in result[-1] )==False and ( 'face' in result[-1] )==False):

            if (result[0] != True and result[-1]!='none' and
                ('n10' in result[-1]) == False and ('face' in result[-1]) == False):
                total_results.append([image_path, json_path, result])


        if (('CLASSIFIER' in folder and is_train==True)or(is_train==False)):
            shuffle(none_class_results)
            none_class_results = none_class_results[:int(len(none_class_results)/2)]
            total_results.extend(none_class_results)
        #if (len(NO_class_results)>0):
        #    number_of_NOs_added += len(NO_class_results)
        #    total_results.extend(NO_class_results)


    print('***'*10)
    print('Total results : ',len(total_results),' Nos added : ',number_of_NOs_added)
    print('\n')


    shuffle(total_results)
    return total_results

# TODO(derekjchow): Add test for pet/PASCAL main files.
def main(_):
  global image_no
  global num_images_in_train
  global num_images_in_test
  global num_images_ignored
  global save_train_filelist
  global save_test_filelist
  global filelist_test

  image_no=0
  save_train_filelist = os.path.join(FLAGS.output_dir, 'signLanguage_train.txt')
  save_test_filelist = os.path.join(FLAGS.output_dir, 'signLanguage_val.txt')
  filelist_test = os.path.join(FLAGS.image_visualization_dir, 'filelist.txt')

  train_folder_filelist = os.path.join(FLAGS.data_dir, 'signLanguage_folder_train.txt')
  test_folder_filelist = os.path.join(FLAGS.data_dir, 'signLanguage_folder_val.txt')

  if os.path.exists(save_train_filelist):   os.remove(save_train_filelist)
  if os.path.exists(save_test_filelist) :   os.remove(save_test_filelist)
  if os.path.exists(filelist_test):       os.remove(filelist_test)

  num_images_in_train = 0
  num_images_in_test = 0
  num_images_ignored = 0

  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
  logging.info('Reading from SignLanguage dataset.')

  encoder_train_dirs = []
  encoder_test_dirs = []

  with open(train_folder_filelist) as f:
      for line in f:
          encoder_train_dirs.append(line[:-1])


  with open(test_folder_filelist) as f:
      for line in f:
          encoder_test_dirs.append(line[:-1])

  data_dir_list = []
  train_dirs = []
  test_dirs = []

  print('Encoder dir train size : ',len(encoder_train_dirs),' test size : ',len(encoder_test_dirs))

  print('Train dirs : ')
  for train_dir in encoder_train_dirs:
      print(train_dir)
  print('Test dirs :  ')
  for test_dir in encoder_test_dirs:
      print(test_dir)
  print('----'*10)

  #exit(0)
  #print(encoder_test_dirs)
  #print(test_dirs)

  train_examples = get_data_from_folders(encoder_train_dirs,is_train=True)
  val_examples = get_data_from_folders(encoder_test_dirs,is_train=False)

  #logging.info('%d training and %d validation examples.',
  #             len(train_examples), len(val_examples))

  train_output_path = os.path.join(FLAGS.output_dir, 'signLanguage_classifier_train_288x352.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'signLanguage_classifier_val_288x352_downsized.record')

  #exit(0)

  create_tf_record(
      train_output_path,
      FLAGS.num_shards,
      label_map_dict,
      train_examples,isTrain=True)

  create_tf_record(
      val_output_path,
      FLAGS.num_shards,
      label_map_dict,
      val_examples)

  print('Num images in train : ',num_images_in_train)
  print('Num images in test  : ',num_images_in_test)
  print('Num images ignored  : ',num_images_ignored)
  print('Created tf records : ')
  print(train_output_path)
  print(val_output_path)
  print(all_class_names,' ',len(all_class_names))
  print(all_sizes)
  print('***'*10)
  print(train_dirs,' ',len(train_dirs))
  print(test_dirs,' ',len(test_dirs))


if __name__ == '__main__':
  tf.app.run()
