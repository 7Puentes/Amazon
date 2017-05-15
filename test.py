import tensorflow as tf
import os

from train import  _generate_image_and_label_batch
class AmazonKaggleRecord:
    pass



def read_input():
    result = AmazonKaggleRecord()

    label_bytes = 1  # 3 ???
    result.height = 256
    result.width = 256
    result.depth = 4

    image_bytes = result.height * result.width * result.depth
    record_bytes = (label_bytes + image_bytes) * 2

    data_dir = '/Users/Charly/Downloads/train-bin-sample'
    filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)

    # filename_queue = tf.train.string_input_producer(['/Users/Charly/Downloads/train-bin-sample/train_10005'])
    filename_queue = tf.train.string_input_producer(filenames)
    result.key, result.value = reader.read(filename_queue)

    # Convert from a string to a vector of uint16 that is record_bytes long.
    bytes = tf.decode_raw(result.value, tf.int16)
    # 262145

    # The first bytes represent the label, which we convert from uint16->int32.
    label_as_bytes = tf.strided_slice(bytes, [0], [label_bytes])
    result.label = tf.cast(label_as_bytes, tf.int16)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    record_as_bytes = tf.strided_slice(bytes, [label_bytes], [label_bytes + image_bytes])
    depth_major = tf.reshape(record_as_bytes, [result.depth, result.height, result.width])

    result.int16image = tf.transpose(depth_major, [1, 2, 0])
    # result.uint16image = depth_major

    # no se para que mierda hace esto
    reshaped_image = tf.cast(result.int16image, tf.float32)

    # Subtract off the mean and divide by the variance of the pixels.
    result.float_image = tf.image.per_image_standardization(reshaped_image)

    # Set the shapes of tensors.
    result.float_image.set_shape([256, 256, 4])
    result.label.set_shape([1])

    return result

if __name__ == '__main__':

    with tf.Session() as session:

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        result = read_input()

        # Generate a batch of images and labels by building up a queue of examples.
        batches = _generate_image_and_label_batch(result.float_image, result.label, 1, 2, shuffle=True)

        mean = tf.reduce_min(batches)

        #print(session.run(tf.shape(bytes)))
        print(session.run(mean))

        coord.request_stop()
        coord.join(threads)