# TODO: Make is better / simpler
# Config file for now
import os

# DATA_PATH = "/Users/ausleihe/Documents/Data/Featurescoop"  # iMac
# DATA_PATH = '/media/gu-ma/Data/Backup-Master-Installation/ausleihe/Documents/Data/Featurescoop'  # UBT
DATA_PATH = "/media/gu-ma/Data/Exchange/Datasets/Featurescoop_salvatore"  # UBT

# FILENAMES
#
SCENES_FN = "scenes.p"
IMAGES_FN = "images.p"
FEATURES_FN = "features.p"
PREDICTIONS_FN = "predictions.p"
#
ALL_SCENES_FN = "all_scenes.p"
ALL_IMAGES_FN = "all_images.p"
ALL_FEATURES_FN = "all_features.p"
ALL_PCA_FEATURES_FN = "all_pca_features.p"
ALL_PREDICTIONS_FN = "all_predictions.p"
#
GRAPH_FN = "graph.p"
PCA_FN = "pca.p"
#
START_FEATURES_FN = "start_features.p"
END_FEATURES_FN = "end_features.p"

# FILEPATH
all_scenes_fp = os.path.join(DATA_PATH, ALL_SCENES_FN)
all_images_fp = os.path.join(DATA_PATH, ALL_IMAGES_FN)
all_features_fp = os.path.join(DATA_PATH, ALL_FEATURES_FN)
all_pca_features_fp = os.path.join(DATA_PATH, ALL_PCA_FEATURES_FN)
all_predictions_fp = os.path.join(DATA_PATH, ALL_PREDICTIONS_FN)
#
pca_fp = os.path.join(DATA_PATH, PCA_FN)
graph_fp = os.path.join(DATA_PATH, GRAPH_FN)

# TODO: Change that shit
start_features_fp = os.path.join(DATA_PATH, START_FEATURES_FN)
end_features_fp = os.path.join(DATA_PATH, END_FEATURES_FN)
#
images_start_folder = os.path.join(DATA_PATH, "00_images_start")
images_end_folder = os.path.join(DATA_PATH, "00_images_end")
# TEMP
poem_index_fp = os.path.join(DATA_PATH, 'poemIndex.ann')
poem_lines_fp = os.path.join(DATA_PATH, 'poemLines.p')
DESCRIPTION_FN = "description.p"
