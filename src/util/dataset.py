import tensorflow as tf
import logging


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def generate(annotations_path, output_path,
             log_step=5000, force_uppercase=True, predict=False):
    logging.info('Building a dataset from {}'.format(annotations_path))
    logging.info('Output file: {}'.format(output_path))

    writer = tf.python_io.TFRecordWriter(output_path)
    with open(annotations_path, 'r') as f:
        for idx, line in enumerate(f):
            line = line.rstrip('\n')

            if not predict:
                # Train/Test
                try:
                    (img_path, label) = line.split('\t', 1)
                except ValueError:
                    logging.error('Missing filename or label, ignoring line {}: {}'.format(idx+1, line))
                    continue

                with open(img_path, 'rb') as img_file:
                    img = img_file.read()

                if force_uppercase:
                    label = label.upper()

                feature = {
                    'image': _bytes_feature(img),
                    'label': _bytes_feature(label.encode('utf-8')),
                    'path': _bytes_feature(img_path.encode('utf-8'))
                }
            else:
                # Batch predict
                img_path = line
                with open(img_path, 'rb') as img_file:
                    img = img_file.read()

                feature = {
                    'image': _bytes_feature(img),
                    'label': _bytes_feature(''.encode('utf-8')),
                    'path': _bytes_feature(img_path.encode('utf-8'))
                }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

            if (idx+1) % log_step == 0:
                logging.info('Processed {} pairs'.format(idx+1))

    logging.info('Dataset is ready: {} pairs'.format(idx+1))
    writer.close()
