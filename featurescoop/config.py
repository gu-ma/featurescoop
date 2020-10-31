# TODO: Make is better / simpler
# Config file for now
import os

# DATA_PATH = "/Users/ausleihe/Documents/Data/Featurescoop"  # iMac
# DATA_PATH = '/media/gu-ma/Data/Backup-Master-Installation/ausleihe/Documents/Data/Featurescoop'  # UBT
DATA_PATH = "/Users/guillaume/Documents/00-App/Datasets/Salvatore/Featurescoop"  # UBT

# FILENAMES
#
SCENES_FN = "scenes.pbz2"
FRAMES_FN = "frames.pbz2"
FEATURES_FN = "features.pbz2"
PCA_FEATURES_FN = "pca_features.pbz2"
PREDICTIONS_FN = "predictions.pbz2"
#
GRAPH_FN = "graph.pbz2"
PCA_FN = "pca.pbz2"
#
START_FEATURES_FN = "start_features.pbz2"
END_FEATURES_FN = "end_features.pbz2"

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
