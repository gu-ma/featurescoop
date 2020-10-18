import os
import argparse
import random

import pickle
import matplotlib.pyplot as plt
from PIL import Image

from config import *
from utils import print_message
from utils import features_get_closest_image
from utils import get_info
from utils import get_scene_info
from utils import graph_get_shortest_paths
from utils import get_concatenated_images

def main(args):

  # Load start and end features 
  img_start, pca_features_start, _ = pickle.load(open(start_features_fp, 'rb'))
  img_end, pca_features_end, _ = pickle.load(open(end_features_fp, 'rb'))
  # 
  img_start = [DATA_PATH+img for img in img_start]
  img_end = [DATA_PATH+img for img in img_end]

  # Load data
  images = pickle.load(open(all_images_fp, 'rb'))
  # 
  images = [DATA_PATH+img for img in images]
  print('Loaded %d images' % len(images))
  pca_features = pickle.load(open(all_pca_features_fp, 'rb'))
  print('Loaded %d pca_features of %d dimensions' % (len(pca_features), len(pca_features[0])))
  predictions = pickle.load(open(all_predictions_fp, 'rb'))
  print('Loaded %d predictions of %d dimensions' % (len(predictions), len(predictions[0])))

  # Load graph
  graph = pickle.load(open(graph_fp, 'rb'))
  print('Loaded graph: %d edges, %d vertices, %d kNN' % (graph.ecount(), graph.vcount(), graph.ecount()/graph.vcount()))

  # Get Random start and end features
  rand1 = int(len(pca_features_start) * random.random())
  rand2 = int(len(pca_features_end) * random.random())
  img1_feat = pca_features_start[rand1]
  img2_feat = pca_features_end[rand2]
  
  # Get some of the closest image id
  print('get closest ids')
  img1_closest_id = features_get_closest_image(pca_features, feature=img1_feat, num_results=args.n)[int(args.n * random.random())]
  img2_closest_id = features_get_closest_image(pca_features, feature=img2_feat, num_results=args.n)[int(args.n * random.random())]

  # img1_closest_id = int(len(pca_features) * random.random())
  # img2_closest_id = int(len(pca_features) * random.random())

  # 
  # print([img1_closest_id, img2_closest_id])
  # 
  print('get shortest paths')
  path_ids = graph_get_shortest_paths(graph, img1_closest_id, img2_closest_id)
  # print(path_ids)
  # Get path to video file
  videos = [ str(get_info(images, id)[1]) for id in path_ids ]
  # print(videos)
  # Get the scenes start and end time
  scenes = [get_scene_info(images, id, max_duration=5)[0:2] for id in path_ids]
  # Create an ordered set (remove the duplicates)
  scenes_details = list(dict.fromkeys(zip(videos, scenes)))
  # subclips = [ VideoFileClip(video).subclip(start, end) for (video, (start, end)) in zip(videos, scenes) ]
  # subclips = [ VideoFileClip(s[0]).subclip(s[1][0], s[1][1]) for s in scenes_details ]
  
  for s in scenes_details:
    print('%s \n%.2f --> %.2d' % (s[0].split('/')[-1], s[1][0], s[1][1]))

  # final_clip = concatenate_videoclips(subclips)
  # final_clip.write_videofile(os.path.join(DATA_PATH, "test.avi"), codec='rawvideo')
  
  img1 = Image.open(img_start[rand1])
  plt.imshow(img1)
  plt.title("img_start (%d)" % rand1)
  plt.show()
  img2 = Image.open(img_end[rand2])
  plt.imshow(img2)
  plt.title("img_end (%d)" % rand2)
  plt.show()
 
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
  


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='TBC')
  parser.add_argument('-n', type=int, help='Number of neighbouring images', default=10)
  args = parser.parse_args()
  main(args) 