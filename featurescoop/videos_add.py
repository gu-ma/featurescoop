# -*- coding: utf-8 -*-
import numpy as np
import argparse
import pathlib 
import shutil
import glob
import re
import os
# 
import pickle
# 
from classifier import ImageClassifier
from models import Models
from config import *
# 
from utils import video_extract_frames
from utils import video_extract_scenes
from utils import video_convert_video
from utils import concatenate_scenes
from utils import print_message


def main(args):

  print_message('Classifying videos from %s' % args.input, type='important', line='new')

  # Init models and classifier
  models = Models()
  classifier = ImageClassifier(models.model, models.feat_extractor)

  videos = glob.glob(args.input+'/**/*.*', recursive=True)

  for i, video in enumerate(videos):

    print_message('Processing video %d of %d \n──%s' % (i+1, len(videos), video), type='processing')

    force = args.force

    # Create folder
    subpath = [p for p in pathlib.Path(video).parts if p not in pathlib.Path(args.input).parts]
    subfolders = subpath[:-1]
    filename = subpath[-1].rsplit('.',1)[0]
    filename = re.sub('[^A-Za-z0-9_]+', '-', filename)
    print(subpath, subfolders, filename)
    path = pathlib.Path(DATA_PATH, args.source, *subfolders, filename)
    os.makedirs(path, exist_ok=True)
    print('Working in directory %s' % path)
    
    # Copy and convert video
    video_fp = os.path.join(path, filename+".mp4")
    if not os.path.isfile(video_fp) or force:
      print('Copying video')
      src = video if args.noconvert else video_convert_video(video, filename)
      shutil.copy(src, video_fp)

    # Extract Scenes
    scenes_fp = os.path.join(path, SCENES_FN)
    if not os.path.isfile(scenes_fp) or force:
      print_message('Extracting scenes')
      scenes = video_extract_scenes(video, args.threshold)
      pickle.dump(scenes, open(scenes_fp, 'wb'))
      force = True
    else:
      scenes = pickle.load(open(scenes_fp, 'rb'))

    # Extract frames
    images_fp = os.path.join(path, IMAGES_FN)
    if not os.path.isfile(images_fp) or force:
      for p in pathlib.Path(path).glob('*.jpg'):
        p.unlink()
      print_message('Extracting frames')
      images = video_extract_frames(video, path, fps=args.fps)
      print_message(
        'Concatenating scenes with %d images per scene (0 = all images)' 
        % args.fpsc
      )
      scenes, count = concatenate_scenes(images, frames_per_scene=args.fpsc)
      images = [images[i] for i in scenes.keys()]
      images_relative = [img.replace(DATA_PATH, '') for img in images]
      pickle.dump(images_relative, open(images_fp, 'wb'))
      print_message(
        'Saved %d images from %d scenes' 
        % (len(images), count)
      )
      force = True
    else:
      images = pickle.load(open(images_fp, 'rb'))

    # Extract features 
    features_fp = os.path.join(path, FEATURES_FN)
    if not os.path.isfile(features_fp) or force:
      print_message('Extracting features from frames')
      features = classifier.extract_features(images)
      pickle.dump(features, open(features_fp, 'wb'))
      print_message(
        'Saved %d features of size %d in a %s' % 
        (features.shape[0], features.shape[1], type(features))
      )
      force = True
    else:
      features = pickle.load(open(features_fp, 'rb'))

    # Extract predictions 
    predictions_fp = os.path.join(path, PREDICTIONS_FN)
    if not os.path.isfile(predictions_fp) or force:
      print_message('Generating predictions from frames')
      predictions = classifier.classify(images)
      pickle.dump(predictions, open(predictions_fp, 'wb'))
      print_message(
        'Saved %d predictions of size %d in a %s' % 
        (len(predictions), len(predictions[0]), type(predictions))
      )
    else:
      predictions = pickle.load(open(predictions_fp, 'rb'))


if __name__ == "__main__":

  # TODO: use docopt
  parser = argparse.ArgumentParser(description='Add new videos')
  parser.add_argument('-i', '--input', type=str, help='Source folder containing the videos to classify', required=True)
  parser.add_argument('-s', '--source', type=str, help='Source (e.g: youtube, vimeo, etc...', required=True)
  parser.add_argument('-fps', '--fps', type=float, help='Frame per sec when extracting frames from videos', default=1)
  parser.add_argument('-fpsc', '--fpsc', type=int, help='Number of frames to extract from each scene (0 = all)', default=0)
  parser.add_argument('-th', '--threshold', type=float, help='Thresold for scenes extraction, low threshold makes short scenes', default=.5)
  parser.add_argument('-nc', '--noconvert', help='Don\'t convert all video files', action='store_true')
  parser.add_argument('-f', '--force', help='Force recreating ALL files', action='store_true')

  args = parser.parse_args()
  main(args) 