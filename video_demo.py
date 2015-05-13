from feat_extractor import FeatExtractor
from caffe_io import load_video
from caffe_io import save_matrix

import os
import sys
import argparse
import time
import caffe
import numpy as np

def main(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument(
    'video_list',
    help = 'Input video list. Put path to video file on each line.')
  parser.add_argument(
    'output_dir',
    help = 'Output directory.')
  parser.add_argument(
    '--sample_rate',
    type = float,
    default = 5.0,
    help = 'Number of frames sampled per second')
  parser.add_argument(
    '--model_def',
    default = '/auto/iris-00/rn/chensun/ThirdParty/caffe_models/vgg_16/VGG_ILSVRC_16_layers_deploy.prototxt',
    help = 'Model definition file (default VGG16)')
  parser.add_argument(
    '--pretrained_model',
    default = '/auto/iris-00/rn/chensun/ThirdParty/caffe_models/vgg_16/VGG_ILSVRC_16_layers.caffemodel',
    help = 'Model parameter file (default VGG16)')
  parser.add_argument(
    '--layers',
    default = 'fc6,fc7',
    help = 'Layers to be extracted, separated by commas')
  parser.add_argument(
    '--cpu',
    action = 'store_true',
    help = 'Use CPU if set')
  parser.add_argument(
    '--oversample',
    action = 'store_true',
    help = 'Oversample 10 patches per frame if set')
  args = parser.parse_args()
  if args.cpu:
    caffe.set_mode_cpu()
    print 'CPU mode'
  else:
    caffe.set_mode_gpu()
    print 'GPU mode'
  oversample = False
  if args.oversample:
    oversample = True
  extractor = FeatExtractor(args.model_def, args.pretrained_model, oversample=oversample)
  blobs = args.layers.split(',')
  with open(args.video_list) as f:
    videos = [l.rstrip() for l in f]
  for video_file in videos:
    frames = load_video(video_file, args.sample_rate)
    if len(frames) < 1: # failed to open the video
      continue
    start = time.time()
    feats = extractor.extract_batch(frames, blobs)
    print '%s feature extracted in %f seconds.' % (os.path.basename(video_file), time.time()-start)
    # save the features
    for blob in blobs:
      feats[blob] = np.array(feats[blob])
    save_matrix(feats, os.path.join(args.output_dir, '%s.mat' % os.path.basename(video_file).split('.')[0]))
  return

if __name__ == '__main__':
  main(sys.argv)