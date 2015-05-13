"""
FeatExtractor is a feature extraction specialization of Net.
"""

import numpy as np
import caffe
from caffe_io import transform_image
import time

class FeatExtractor(caffe.Net):
  """
  Calls caffe_io to convert video/images
  and extract embedding features
  """
  def __init__(self, model_file, pretrained_file, img_dim = 256,
               crop_dim = 224, mean = [103.939, 116.779, 123.68], oversample = False):
    caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)
    self.img_dim = img_dim
    self.crop_dim = crop_dim
    self.mean = mean
    self.oversample = oversample
    self.batch_size = 10  # hard coded, same as oversample patches

  def extract(self, imgs, blobs = ['fc6', 'fc7']):
    feats = {}
    for blob in blobs:
      feats[blob] = []
    for img in imgs:
      data = transform_image(img, self.oversample, self.mean, self.img_dim, self.crop_dim)
      # Use forward all to do the padding
      out = self.forward_all(**{self.inputs[0]: data, 'blobs': blobs})
      for blob in blobs:
        feat = out[blob]
        if self.oversample:
          feat = feat.reshape((len(feat) / self.batch_size, self.batch_size, -1))
          feat = feat.mean(1)
        feats[blob].append(feat.flatten())
    return feats

  def _process_batch(self, data, feats, blobs):
    if data is None:
      return
    out = self.forward_all(**{self.inputs[0]: data, 'blobs': blobs})
    for blob in blobs:
      feat = out[blob]
      feat = feat.reshape((len(feat) / data.shape[0], data.shape[0], -1))
      for i in xrange(data.shape[0]):
        feats[blob].append(feat[:, i, :].flatten())

  def extract_batch(self, imgs, blobs = ['fc6', 'fc7']):
    if self.oversample:   # Each oversampled image is a batch
      return self.extract(imgs, blobs)
    feats = {}
    for blob in blobs:
      feats[blob] = []
    data = None
    for img in imgs:
      if data is None:
        data = transform_image(img, self.oversample, self.mean, self.img_dim, self.crop_dim)
      else:
        data = np.vstack((data, transform_image(img, self.oversample, self.mean, self.img_dim, self.crop_dim)))
      if data.shape[0] == self.batch_size:
        self._process_batch(data, feats, blobs)
        data = None
    self._process_batch(data, feats, blobs)
    return feats

if __name__ == '__main__':
  caffe.set_mode_gpu()
  model_file = './caffe_models/vgg_16/VGG_ILSVRC_16_layers_deploy.prototxt'
  pretrained_file = './caffe_models/vgg_16/VGG_ILSVRC_16_layers.caffemodel'
  img_list = 'flickr8k_images_fullpath.lst'
  start = time.time()
  extractor = FeatExtractor(model_file, pretrained_file, oversample=False)
  print 'intitialization time:', time.time() - start
  from caffe_io import load_image

  with open(img_list) as f:
    img_names = [l.rstrip() for l in f]
  imgs = []
  for i in range(13):
    img_name = img_names[i]
    img = load_image(img_name)
    imgs.append(img)
  start = time.time()
  feats1 = extractor.extract(imgs)
  print 'non-batch extraction:', time.time() - start
  start = time.time()
  feats2 = extractor.extract_batch(imgs)
  print 'batch extraction:', time.time() - start

  print len(feats1['fc6']), len(feats2['fc6'])
  for i in xrange(len(feats1['fc6'])):
    print feats1['fc6'][i].shape
    print (feats1['fc6'][i]==feats2['fc6'][i]).all()
