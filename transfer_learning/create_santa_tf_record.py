import PIL
import io
import csv
import tensorflow as tf

from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', 'santa_train.record', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def create_tf_example(example):
  with open(example['filename'], 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  width, height = image.size
  filename = example['filename'].encode('utf8') # Filename of the image. Empty if image is not from file
  encoded_image_data = encoded_jpg # Encoded image bytes
  image_format = b'jpeg' # b'jpeg' or b'png'

  xmins = [float(example['xmin'])] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [float(example['xmax'])] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [float(example['ymin'])] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [float(example['ymax'])] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = ["Santa".encode('utf8')] # List of string class name of bounding box (1 per box)
  classes = [1] # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  csvfile = open('santa.csv', 'r')
  column_names = ("filename", "xmin", "xmax", "ymin", "ymax")
  examples = csv.DictReader(csvfile, column_names)

  for example in examples:
    tf_example = create_tf_example(example)
    writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()
