import argparse
import random
import os

import pickle

import matplotlib.pyplot as plt

# from moviepy.editor import VideoFileClip, concatenate_videoclips

from classifier import ImageClassifier
from models import Models
from config import *

from utils import features_apply_pca
from utils import features_get_closest_image

from utils import get_scene_info
from utils import get_info

from utils import graph_get_shortest_paths

from utils import get_concatenated_images
from utils import print_message


def test_closest(pca_features, images):
  # 
  query_img_id = int(len(pca_features) * random.random())
  id_closest = features_get_closest_image(pca_features, id=query_img_id, num_neighbors=10)
  query_img = get_concatenated_images(images, [query_img_id], 300)
  closest_img = get_concatenated_images(images, id_closest, 200)
  # 
  plt.imshow(query_img)
  plt.title("query image (%d)" % query_img_id)
  plt.show()
  # 
  plt.figure(figsize = (20,10))
  plt.imshow(closest_img)
  plt.title("closest images")
  plt.show()


def test_path(pca_features, images, pca_fp, classifier, graph, DATA_PATH, img1='', img2=''):
  if img1 and img2:
    # Extract Features
    print('Extracting features')
    imgs_feat = classifier.extract_features([img1, img2], batch_size=64)
    # Reduce dimensions with PCA 
    print('Applying PCA')
    pca = pickle.load(open(pca_fp, 'rb'))
    imgs_pca_feat = features_apply_pca(imgs_feat, pca=pca)[1]
    # Get one of the closest image id
    img1_closest_id = features_get_closest_image(pca_features, feature=imgs_pca_feat[0], num_neighbors=10)[int(10 * random.random())]
    img2_closest_id = features_get_closest_image(pca_features, feature=imgs_pca_feat[1], num_neighbors=10)[int(10 * random.random())]
  else:
    img1_closest_id = int(len(pca_features) * random.random())
    img2_closest_id = int(len(pca_features) * random.random())
  # 
  print([img1_closest_id, img2_closest_id])
  path_ids = graph_get_shortest_paths(graph, img1_closest_id, img2_closest_id)
  print(path_ids)
  # Get path to video file
  videos = [ str(get_info(images, id)[1]) for id in path_ids ]
  print(videos)
  # Get the scenes start and end time
  scenes = [get_scene_info(images, id, max_duration=5)[0:2] for id in path_ids]
  # Create an ordered set (remove the duplicates)
  scenes_details = list(dict.fromkeys(zip(videos, scenes)))
  # subclips = [ VideoFileClip(video).subclip(start, end) for (video, (start, end)) in zip(videos, scenes) ]
  subclips = [ VideoFileClip(s[0]).subclip(s[1][0], s[1][1]) for s in scenes_details ]
  
  for s in scenes_details:
    print('%s \n%.2f --> %.2d' % (s[0].split('/')[-1], s[1][0], s[1][1]))

  final_clip = concatenate_videoclips(subclips)
  final_clip.write_videofile(os.path.join(DATA_PATH, "test.avi"), codec='rawvideo')

  #
  imgs = get_concatenated_images(images, [img1_closest_id, img2_closest_id], 300)
  plt.imshow(imgs)
  plt.title("img1 (%d) img2 (%d)" % (img1_closest_id, img2_closest_id))
  plt.show()
 
  # retrieve the images, concatenate into one, and display them
  results_image = get_concatenated_images(images, path_ids, 200)
  plt.figure(figsize = (16,12))
  plt.imshow(results_image)
  plt.show()


def main(args):

  # Test(s)
  images = pickle.load(open(all_images_fp, 'rb'))  
  pca_features = pickle.load(open(all_pca_features_fp, 'rb'))
  predictions = pickle.load(open(all_predictions_fp, 'rb'))
  graph = pickle.load(open(graph_fp, 'rb'))
  print('Loaded %d images' % len(images))
  print('Loaded %d pca_features of %d dimensions' % (len(pca_features), len(pca_features[0])))
  print('Loaded %d predictions of %d dimensions' % (len(predictions), len(predictions[0])))
  print('Loaded graph: %d edges, %d vertices, %d kNN' % (graph.ecount(), graph.vcount(), graph.ecount()/graph.vcount()))
  
  images = [DATA_PATH+img for img in images]

  if args.test == 'closest':
    print_message('Testing closest distance', line='new')
    test_closest(pca_features, images)
  elif args.test == 'path':
    # Init models and classifier
    models = Models()
    classifier = ImageClassifier(models.model, models.feat_extractor)    
    print_message('Testing images path', line='new')
    test_path(pca_features, images, pca_fp, classifier, graph, DATA_PATH, args.img1, args.img2)


if __name__ == "__main__":

  # TODO: use docopt
  parser = argparse.ArgumentParser(description='Tests')
  parser.add_argument('-t', '--test', type=str, help='Test mode', default='path', choices=['path', 'closest'])
  parser.add_argument('-i1','--img1', type=str, help='From image (test path)')
  parser.add_argument('-i2','--img2', type=str, help='To image (test path)')

  args = parser.parse_args()
  main(args) 