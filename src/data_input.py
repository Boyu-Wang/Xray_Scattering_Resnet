# read binary data

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import tensorflow as tf

IMAGE_SIZE = 224

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 500*200*0.8
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 500*200*0.2


def read_data(filename_queue, num_tags):
  class DataRecord(object):
    pass
  result = DataRecord()

  result.height = 256
  result.width = 256
  result.depth = 1

  label_bytes = num_tags
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = (label_bytes + image_bytes) * 2  # int16 encoding

  # Read a record, getting filenames from the filename_queue. 
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.int16)

  result.label = tf.cast(
      tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

  depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                           [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.float32image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle, num_preprocess_threads=16):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 1] of type.float32.
    label: 1-D Tensor of [NUM_TAGS] of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 1] size.
    labels: Labels. 2D tensor of [batch_size, NUM_TAGS] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.

  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
      [image, label],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=min_queue_examples + 30 * batch_size,
      min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
      [image, label],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=800)

  # Display the training images in the visualizer.
  tf.summary.image('images', images, max_outputs=20)

  return images, label_batch


def distorted_inputs(data_dir, batch_size, num_tags=17):
  """Construct distorted input for training using the Reader ops.

  Args:
    data_dir: Path to the data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
    labels: Labels. 2D tensor of [batch_size, NUM_TAGS] size.
  """

  filenames = [os.path.join(data_dir, 'train_batch_%d.bin' % i)
               for i in range(0, 8)]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)
  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_data(filename_queue,num_tags=num_tags)
  reshaped_image = read_input.float32image

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # reshape image
  distorted_image = tf.image.resize_images(reshaped_image, np.array([height, width]))
  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(distorted_image)
  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)


def inputs(data_dir, batch_size, num_tags=17):
  """Construct input for evaluation using the Reader ops.

  Args:
    data_dir: Path to the data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
    labels: Labels. 1D tensor of [batch_size, NUM_TAGS] size.
  """
  num_preprocess_threads = 16
  filenames = [os.path.join(data_dir, 'val_batch_%d.bin' % (i))
                 for i in range(0, 2)]
  num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)
  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_data(filename_queue, num_tags=num_tags)
  reshaped_image = read_input.float32image

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  resized_image = tf.image.resize_images(reshaped_image, np.array([height, width]))
  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(resized_image)
  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False, num_preprocess_threads=num_preprocess_threads)
