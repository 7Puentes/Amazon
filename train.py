import os
import datetime , time

import amazon
from log import LoggerHook
from settings import *

import tensorflow as tf

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])



def read_input():

    data_dir = '/Users/Charly/Downloads/train-bin-sample'

    filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    for f in filenames:
        print f
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    print "Entrenando con %i" % len(filenames)

    filename_queue = tf.train.string_input_producer(filenames)

    class AmazonKaggleRecord(object):
        pass


    result = AmazonKaggleRecord()

    label_bytes = 1 # 3 ???
    result.height = 256
    result.width = 256
    result.depth = 4

    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = (label_bytes + image_bytes)*2

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint16 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.int16)

    # The first bytes represent the label, which we convert from uint16->int32.
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int16)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes],[label_bytes + image_bytes]), [result.depth, result.height, result.width])

    # Convert from [depth, height, width] to [height, width, depth].
    result.int16image = tf.transpose(depth_major, [1, 2, 0])
    #result.uint16image = depth_major

    # no se para que mierda hace esto
    reshaped_image = tf.cast(result.int16image, tf.float32)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(reshaped_image)

    # Set the shapes of tensors.
    float_image.set_shape([256, 256, 4])
    result.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, result.label, min_queue_examples, FLAGS.batch_size, shuffle=True)


def train():

  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    images, labels = read_input()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = amazon.inference(images)

    # Calculate loss.
    loss = amazon.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = amazon.train(loss, global_step)

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps), tf.train.NanTensorHook(loss), LoggerHook(loss)],
        config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) as mon_sess:

      while not mon_sess.should_stop():
        mon_sess.run(train_op)

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)

    tf.app.run()
