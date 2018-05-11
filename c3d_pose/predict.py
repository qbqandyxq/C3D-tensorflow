import os.path
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import c3d_model
import numpy as np
import show as sh
dataset_dir = '/home/qbq/Documents/data/UCF-101/'

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 1
flags.DEFINE_integer('batch_size', 1 , 
                     'important, batch_size place size to 1 only')

flags.DEFINE_string('checkpoint', 
                    '/home/qbq/Documents/code/C3D-tensorflow/work_class3/c3d_ucf_model-1999',
                    "the path to checkpoint saved dir")
flags.DEFINE_string('TEST_LIST_PATH',
                    '/home/qbq/Desktop/c3d_pose/list/test3.list',
                    "the path to test.list dir")
flags.DEFINE_string('write_to_txt',
                    '/home/qbq/Desktop/c3d_pose/output/work.txt',
                    'after do the prediction op, the message will write to the path save as a txt')
FLAGS = flags.FLAGS

test_num = input_data.get_test_num(FLAGS.TEST_LIST_PATH)

def placeholder_inputs(batch_size):
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS))
  labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
  return images_placeholder, labels_placeholder

def _variable_on_cpu(name, shape, initializer):
  #with tf.device('/cpu:%d' % cpu_id):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var) * wd
    tf.add_to_collection('losses', weight_decay)
  return var

def run_test():
  #model_name = "./sports1m_finetuning_ucf101.model"
  model_name = FLAGS.checkpoint
  test_list_file = FLAGS.TEST_LIST_PATH
  num_test_videos = len(list(open(test_list_file,'r')))
  print(("Number of test videos={}".format(num_test_videos)))

  # Get the sets of images and labels for training, validation, and
  images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size * gpu_num)
  with tf.variable_scope('var_name') as var_scope:
    weights = {
            'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
            'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
            'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
            'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
            'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
            'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
            'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
            'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.04, 0.005)
            }
    biases = {
            'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
            'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
            'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
            'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
            'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
            'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
            'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
            'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
            'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
            'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0),
            'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.04, 0.0),
            }
  logits = []
  for gpu_index in range(0, gpu_num):
    with tf.device('/gpu:%d' % gpu_index):
      logit = c3d_model.inference_c3d(images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:,:,:,:], 0.6, FLAGS.batch_size, weights, biases)
      logits.append(logit)
  logits = tf.concat(logits,0)
  norm_score = tf.nn.softmax(logits)
  
  saver = tf.train.Saver()
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  init = tf.global_variables_initializer()
  sess.run(init)
  # Create a saver for writing training checkpoints.
  print("load model:", model_name)
  saver.restore(sess, model_name)
  # And then after everything is built, start the training loop.
  bufsize = 4
  write_file = open(FLAGS.write_to_txt, "w+", bufsize)
  next_start_pos = 0
  all_steps = int((num_test_videos - 1) / (FLAGS.batch_size * gpu_num) + 1)
  loss=[]
  accuracy_epoch = 0
  accuracy_out=0
  start_time = time.time()
  lines = open(test_list_file,'r')
  #for step in range(all_steps):
  start=0
  for line in lines:   
    line = line.strip('\n').split()
    dataset_dir=line[0]
    infer=[]
    true_label=line[1]
    start +=1
    test_num=0
    print("  ", start)
    print("infer ", dataset_dir)
    for i in range(len(os.listdir(dataset_dir))//c3d_model.NUM_FRAMES_PER_CLIP):
        
        test_images, test_labels, next_start_pos, or_dir, valid_len = \
            input_data.read_clip_and_label(
                    dataset_dir,
                    test_list_file,
                    FLAGS.batch_size * gpu_num,
                    start_pos=i,
                    )
        predict_score = norm_score.eval(
            session=sess,
            feed_dict={images_placeholder: test_images}
            )
        true_label = true_label
        top1_predicted_label = np.argmax(predict_score)
        infer.append(top1_predicted_label)
    write_file.write('{}|{}|{}\n'.format(
              true_label,
              dataset_dir,
              infer,
              ))

  write_file.close()
  end_time = time.time()
  print("done")
  print("total time for test: ", (end_time - start_time))
  print("result writing to :", FLAGS.write_to_txt)

def main(_):
  run_test()

if __name__ == '__main__':
  tf.app.run()
