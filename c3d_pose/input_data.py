import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
from PIL import ImageEnhance
import random
import numpy as np
import cv2
import time
#def get_a_gs_random():
    #res_from = np.random.normal(1. , 0.5, 50)
 #   choic = np.random.choice([0.3, 0.5, 0.7, 1, 1, 1, 1.3, 1.5, 2])
    #choic = np.random.choice(res_from)
    #if choic < 0.3:
     #   choic = 0.3
  #  return choic
def get_test_num(filename):
    #print("asdfsdf",filename)
    lines = open(filename, 'r')
    return len(list(lines))
def get_frames_data(file_name_dir, num_frames_per_clip=16, data_augmentation=False, start=0):
  ''' Given a directory containing extracted frames, return a video clip of
  (num_frames_per_clip) consecutive frames as a list of np arrays '''
  ret_arr = []
  s_index = 0
  #print("filename", filename)
  for parent, dirnames, filenames in os.walk(file_name_dir):
    if(len(filenames)<num_frames_per_clip):
      return [], s_index
    filenames = sorted(filenames)
    
    #s_index = random.randint(0, len(filenames) - num_frames_per_clip)
    s_index = start
    brit = 1
    if data_augmentation:
      brit = np.random.choice([0.3, 0.5, 0.7, 1, 1, 1, 1.3, 1.5, 2])
    for i in range(s_index*num_frames_per_clip, s_index*num_frames_per_clip + num_frames_per_clip):
      image_name = str(file_name_dir) + '/' + str(filenames[i])
      #print(image_name)
      img = Image.open(image_name)
      imgEH = ImageEnhance.Brightness(img).enhance(brit)
      img_data = np.array(imgEH)
      ret_arr.append(img_data)
  return ret_arr, s_index

def read_clip_and_label(dataset_dir, filename, batch_size, start_pos=-1, num_frames_per_clip=16, crop_size=112, shuffle=False, data_augmentation=False):
  lines = open(filename,'r')
  read_dirnames = []
  data = []
  label = []
  batch_index = 0
  next_batch_start = -1
  lines = list(lines)
  np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])

  video_indices = list(range(start_pos, len(lines)))
  for index in video_indices:
    if(batch_index>=batch_size):
      next_batch_start = index
      break
    #print("==================",dataset_dir)
    line = lines[index].strip('\n').split()
    dirname = line[0]
   
    #print(dirname)
    tmp_label = line[1]
    
    #if not shuffle:
      #print("Loading a video clip from {}...".format(dirname))
    
    #tmp_data, _ = get_frames_data(dirname, num_frames_per_clip, data_augmentation, start=start_pos)
    tmp_data, _ = get_frames_data(dataset_dir, num_frames_per_clip, data_augmentation, start=start_pos)
    img_datas = [];
    if(len(tmp_data)!=0):
      for j in range(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))

        if(img.width>img.height):
          scale = float(crop_size)/float(img.height)
          img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), crop_size))).astype(np.float32)
        else:
          scale = float(crop_size)/float(img.width)
          img = np.array(cv2.resize(np.array(img),(crop_size, int(img.height * scale + 1)))).astype(np.float32)
        crop_x = int((img.shape[0] - crop_size)/2)
        crop_y = int((img.shape[1] - crop_size)/2)
        img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:] - np_mean[j]
        img_datas.append(img)
      data.append(img_datas)
      label.append(int(tmp_label))
      batch_index = batch_index + 1
      read_dirnames.append(dirname)
  # data is 0. does not get the data.

  # pad (duplicate) data/label if less than batch_size
  valid_len = len(data)
  pad_len = batch_size - valid_len
  if pad_len:
    for i in range(pad_len):
      data.append(img_datas)
      label.append(int(tmp_label))

  np_arr_data = np.array(data).astype(np.float32)
  np_arr_label = np.array(label).astype(np.int64)
  #print("np_arr_data", np_arr_data.shape)
  return np_arr_data, np_arr_label, next_batch_start, read_dirnames, valid_len
