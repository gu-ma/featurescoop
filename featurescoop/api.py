import os
import random
from time import sleep

import pickle
from flask import Flask, request
from flask_restful import Resource, Api
from flask import jsonify
from json import dumps

# TODO: Remove that part or make it properly
import spacy
import numpy as np
from annoy import AnnoyIndex

from config import *
from utils import features_get_closest_image
from utils import get_info
from utils import get_scene_info
from utils import graph_get_shortest_paths

app = Flask(__name__)
api = Api(app)

def meanvector(text):
  s = nlp(text)
  vecs = [word.vector for word in s \
            if word.pos_ in ('NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'ADP') \
            and np.any(word.vector)] # skip all-zero vectors
  if len(vecs) == 0:
    raise IndexError
  else:
    return np.array(vecs).mean(axis=0)

def loadIndex():
  t = AnnoyIndex(384)
  t.load(poem_index_fp) # super fast, will just mmap the file
  l = pickle.load(open(poem_lines_fp, 'rb'))
  return t, l

def retrievePath(num_neighbors, start_id, order):

  if order == 'StartToEnd':

    if start_id == -1:
      img1_feat = pca_features_start[int(len(pca_features_start) * random.random())]
    else:
      img1_feat = pca_features[start_id]
    img2_feat = pca_features_end[int(len(pca_features_end) * random.random())]

  if order == 'EndToStart':

    if start_id == -1:
      img1_feat = pca_features_end[int(len(pca_features_end) * random.random())]
    else:
      img1_feat = pca_features[start_id]
    img2_feat = pca_features_start[int(len(pca_features_start) * random.random())]

  # OPTIMIZE
  # ??
  # https://stackoverflow.com/questions/36422949/use-pdist-in-python-with-a-custom-distance-function-defined-by-you 
  img1_id = features_get_closest_image(
    pca_features, feature=img1_feat, num_neighbors=num_neighbors
    )[int(num_neighbors * random.random())]
  
  img2_id = features_get_closest_image(
    pca_features, feature=img2_feat, num_neighbors=num_neighbors
    )[int(num_neighbors * random.random())]

  path_ids = graph_get_shortest_paths(graph, img1_id, img2_id)

  return path_ids


def sbuildScenesJson(img_ids, duration):

  videos = [ str(get_info(images, id)[1]) for id in img_ids ]

  scenes_timing = [get_scene_info(images, id, max_duration=duration)[0:2] for id in img_ids]
  scenes_ids = [get_scene_info(images, id, max_duration=duration)[3:5] for id in img_ids]

  # TODO: Remove that part or make it properly
  # This is very ugly
  descriptions = [ pickle.load(open(os.path.join(str(get_info(images, id)[2]), DESCRIPTION_FN), 'rb')) if os.path.isfile(os.path.join(str(get_info(images, id)[2]), DESCRIPTION_FN)) else '' for id in img_ids]

  # Remove duplicates
  # TODO: remove all sequences from the same videos
  scenes_details = list(dict.fromkeys(zip(videos, scenes_timing, scenes_ids, descriptions)))
  # scenes_details = list(dict.fromkeys(zip(videos, scenes_timing, scenes_ids)))
  # scenes_details = dict.fromkeys(zip(videos, scenes_timing, scenes_ids))

  # sleep(2)

  t = []
  for s in scenes_details:
    a = {}
    a['file'] = s[0]
    a['start_time'] = s[1][0]
    a['end_time'] = s[1][1]
    a['start_id'] = s[2][0]
    a['end_id'] = s[2][1]
    # TODO: Remove that part or make it properly
    a['description'] = s[3]
    a['poem'] = lines[index.get_nns_by_vector(meanvector(s[3]), n=5)[int(random.random()*5)]] if s[3] != '' else ''
    # 
    t.append(a)

  # print(list(scenes_details))
  result = {'scenes': t}

  return result


class Getpath(Resource):
  def get(self):
    
    num_neighbors = (
      int(request.args.get('num_neighbors'))
      if request.args.get('num_neighbors') 
      else 10
    )

    duration = (
      float(request.args.get('duration'))
      if request.args.get('duration') 
      else 5.0
    )

    order = (
      str(request.args.get('order'))
      if request.args.get('order') 
      else 'StartToEnd'
    )

    start_id = (
      int(request.args.get('start_id'))
      if request.args.get('start_id') 
      else -1
    )

    min_length = (
      int(request.args.get('min_length'))
      if request.args.get('min_length') 
      else 2
    )

    img_ids = retrievePath(num_neighbors, start_id, order)

    while len(img_ids) < min_length:
      img_ids = retrievePath(num_neighbors, start_id, order)

    result = buildScenesJson(img_ids, duration)

    return jsonify(result)


class GetSimilarVideos(Resource):
  def get(self):
    
    num_neighbors = (
      int(request.args.get('num_neighbors'))
      if request.args.get('num_neighbors') 
      else 10
    )

    duration = (
      float(request.args.get('duration'))
      if request.args.get('duration') 
      else 5.0
    )

    id = (
      int(request.args.get('id'))
      if request.args.get('id') 
      else -1
    )

    if id == -1:

      img1_feat = pca_features_start[int(len(pca_features_start) * random.random())]
      id = features_get_closest_image(
        pca_features, feature=img1_feat, num_neighbors=num_neighbors
        )[int(num_neighbors * random.random())]

    img_ids = features_get_closest_image(pca_features, id=id, num_neighbors=num_neighbors)

    while len(img_ids) == 0:
      img_ids = retrievePath(num_neighbors, start_id, order)


    result = buildScenesJson(img_ids, duration)

    return jsonify(result)

 
api.add_resource(Getpath, '/getpath')
api.add_resource(GetSimilarVideos, '/getsimilarvideos')


if __name__ == '__main__':

  # TODO: Remove that part or make it properly 
  nlp = spacy.load('en')
  index, lines = loadIndex();

  img_start, pca_features_start, _ = pickle.load(open(start_features_fp, 'rb'))
  img_end, pca_features_end, _ = pickle.load(open(end_features_fp, 'rb'))

  images = pickle.load(open(all_images_fp, 'rb'))
  images = [DATA_PATH+img for img in images]

  pca_features = pickle.load(open(all_pca_features_fp, 'rb'))
  predictions = pickle.load(open(all_predictions_fp, 'rb'))
  graph = pickle.load(open(graph_fp, 'rb'))

  app.config["JSON_SORT_KEYS"] = False
  app.run(host='0.0.0.0', port=5002)
