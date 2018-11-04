import os 
import sys
import tarfile
import numpy
from six.moves import urllib

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import numpy as np

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = "/opt/cifar10"
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  #extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
  extracted_dir_path = os.path.join(dest_directory, 'train')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def read_cifar10(filename_queue):
  """Reads and parses examples from CIFAR10 data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  label_bytes = 1  # 2 for CIFAR-100
  result.height = 32
  result.width = 32
  result.depth = 3
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),
      [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result

def inputs(eval_data, data_dir, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not eval_data:
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames= [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
 
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input=read_cifar10(filename_queue)
  return read_input


def mergeFilesIntoFile(data_dir,outTrainFile, outTestFile):
  filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
	       for i in xrange(1,6)]
  #for i in xrange(0,5):
  #  print(filenames[i])
  fileNameOut = os.path.join(data_dir,outTrainFile)
  #print(fileNameOut)
  fout = open(fileNameOut,'wb')
  for i in xrange(0,5):
    fin=open(filenames[i],'rb')
    while True:
      data=fin.read(65536)
      if not data:
        break
      fout.write(data)
    fin.close()
  fout.close()

IM_SIZE = 32

def extractImageDataset(fileName):
  print('Extracting Dataset...')
  print(fileName)
  statinfo = os.stat(fileName)
  filesize = statinfo.st_size
  setCnt = int(filesize / (IM_SIZE * IM_SIZE * 3 + 1))
  print('Dataset Count : ',setCnt)
  fin = open(fileName,'rb')
  #labels = []
  #images = []

  npLabels = np.zeros(setCnt,np.int8)
  npImages = np.zeros((setCnt,IM_SIZE * IM_SIZE * 3),np.int8)
  cnt = 0
  while True:
    data = fin.read(1)
    if not data:
      break
    npLabels[cnt] = np.frombuffer(data,dtype=np.int8)
    #npLabels[cnt] = data
    buf = fin.read(IM_SIZE * IM_SIZE * 3)
    npImages[cnt,:] = np.frombuffer(buf,dtype=np.int8) 
    cnt = cnt + 1
  

  #for i in xrange(0,19):
  #  print('Label ',int(labels[i]),' index ', i)
  #npLabels = np.array(labels,dtype='uint8')
  #npImages = np.frombuffer(images,dtype='int8')
  
  npImages = npImages.reshape(cnt, IM_SIZE * IM_SIZE, 3)
  npImages = npImages.astype(np.float32)
  npImages = np.multiply(npImages, 1.0 / 255.0)
  
  return npLabels,npImages



class Cifar10Dataset(object):
  def __init__(self,
	       images,
               labels):
    self._images = images
    self._labels = labels
    self._num_examples = len(images)
  def images(self):
    return self._images
  def labels(self):
    return self._labels
  def num_examples(self):
    return self._num_examples  
