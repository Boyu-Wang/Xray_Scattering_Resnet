from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import os

import data_input
from convert_to_tfrecords import tags_meta

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '../save/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 5000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('data_dir', '../data/tfrecords',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_integer('startep', 0,'starting epoch, 0 start from scratch')
tf.app.flags.DEFINE_integer('gpuid', 0,'gpuid')
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")

import resnet_model as model

train_dir = os.path.join(FLAGS.train_dir, 'resnet')
num_tags = len(tags_meta)
num_training_images = 80000


def train():
  os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpuid)
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels
    images, labels = data_input.distorted_inputs(data_dir=FLAGS.data_dir,
            batch_size=FLAGS.batch_size, num_tags=num_tags)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = model.inference(images, is_training=True, num_classes=num_tags)

    # Calculate loss.
    loss = model.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = model.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.log_device_placement=FLAGS.log_device_placement
    config.gpu_options.allow_growth = True
    # Start running operations on the Graph.
    sess = tf.Session(config=config)
    sess.run(init)

    step_init = 0

    if FLAGS.startep > 0:
      ckpt = tf.train.get_checkpoint_state(train_dir)
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        print('load from pretrained model')
        saver.restore(sess, ckpt.model_checkpoint_path)
        # extract global_step from it.
        step_init = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
      else:
        print('No checkpoint file found')
        return
    else:
      print('random initialize the model')

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
    step_per_epoch = num_training_images / FLAGS.batch_size
    print('step per epoch: %d' % step_per_epoch)
    for step in np.arange(step_init, FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % (step_per_epoch/2) == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % step_per_epoch == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % step_per_epoch == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
        epoch_num = int(step / step_per_epoch)
        saver.save(sess, checkpoint_path, global_step=epoch_num)


def main(argv=None):  # pylint: disable=unused-argument
  if FLAGS.startep <= 0:
    if tf.gfile.Exists(train_dir):
      tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
  print('writing every thing to %s' % train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
