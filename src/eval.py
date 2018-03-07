from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import sklearn.metrics as smetrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import scipy.io as sio
import numpy as np
import tensorflow as tf
import os
import progressbar

import data_input
from convert_to_tfrecords import tags_meta

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '../save/',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../save/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 20000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_integer('eval_batch_size', 100,
                            """Number of batch size to evaluate.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_integer('gpuid', 0,'gpuid')
tf.app.flags.DEFINE_string('data_dir', '../data/tfrecords',
                           """Path to the data directory.""")

import resnet_model as model

checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, 'resnet')
eval_dir = os.path.join(FLAGS.eval_dir, 'eval')

num_tags = len(tags_meta)
num_eval_images = 20000


def eval_once(saver, summary_writer, prob_op, gt_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    prob_op: prob op.
    gt_op: ground truth op.
    summary_op: Summary op.
  """
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  with tf.Session(config=config) as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      print('load from pretrained model from')
      print(ckpt.model_checkpoint_path)
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(num_eval_images / FLAGS.eval_batch_size))
      num_examples = num_iter * FLAGS.eval_batch_size
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_examples * num_tags
      pred_prob_all = np.zeros([num_examples, num_tags])
      gt_all = np.zeros([num_examples, num_tags])
      loss = 0
      step = 0
      bar = progressbar.ProgressBar(maxval=num_iter)
      bar.start()
      while step < num_iter and not coord.should_stop():
        bar.update(step)
        pred_prob, gt_label = sess.run([prob_op, gt_op])

        pred_prob_all[step*FLAGS.eval_batch_size : (step+1)*FLAGS.eval_batch_size,:] = pred_prob
        gt_all[step*FLAGS.eval_batch_size : (step+1)*FLAGS.eval_batch_size,:] = gt_label
        true_count += ((pred_prob > 0.5) == gt_label.astype('bool')).sum()

        step += 1
      bar.finish()

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision = %.3f' % (datetime.now(), precision))

      # Compute mean averaged precision
      gt_all_bool = gt_all.astype('bool')
      aps = smetrics.average_precision_score(gt_all_bool, pred_prob_all, average=None)
      for c_i in range(num_tags):
        print('%s:\t%.3f'%(tags_meta[c_i][1], aps[c_i]))
      meanAP = np.nanmean(aps)
      print('%s: mAP = %.3f' % (datetime.now(), meanAP))
      pred_metrics = dict()
      pred_metrics['gt'] = gt_all
      pred_metrics['pred'] = pred_prob_all
      pred_metrics['aps'] = aps
      pred_metrics['mAP'] = meanAP
      sio.savemat(os.path.join(eval_dir, 'pred.mat'), pred_metrics)

      # plot ROC curve
      fpr = dict()
      tpr = dict()
      roc_auc = dict()
      for class_idx in range(num_tags):
        fpr[class_idx], tpr[class_idx], _ = roc_curve(gt_all[:,class_idx], pred_prob_all[:,class_idx])
        roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])
        plt.figure()
        plt.plot(fpr[class_idx], tpr[class_idx], label='ROC curve (area = %0.2f)' % roc_auc[class_idx])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic class %s'% tags_meta[class_idx][1])
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(eval_dir, 'roc_%d_%s.png' % (class_idx, tags_meta[class_idx][1])))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision', simple_value=precision)
      summary.value.add(tag='mAP', simple_value=meanAP)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

  return meanAP


def evaluate():
  os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpuid)
  with tf.Graph().as_default() as g:
    images, labels = data_input.inputs(data_dir=FLAGS.data_dir,batch_size=FLAGS.eval_batch_size,num_tags=num_tags)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = model.inference(images, is_training=False, num_classes=num_tags)

    # Calculate predictions.
    prob_op = tf.sigmoid(logits)

    # Restore the moving average version of the learned variables for eval.
    saver = tf.train.Saver(tf.global_variables())

    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(eval_dir, g)

    while True:
      eval_once(saver, summary_writer, prob_op, labels, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(eval_dir):
    tf.gfile.DeleteRecursively(eval_dir)
  tf.gfile.MakeDirs(eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
