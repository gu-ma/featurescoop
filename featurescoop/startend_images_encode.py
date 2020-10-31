import os
import glob
import argparse
from distutils.dir_util import copy_tree
from distutils.dir_util import remove_tree

from tqdm import tqdm
import pickle

from utils import print_message
from utils import features_apply_pca

#
from config import *

#
from classifier import ImageClassifier
from models import Models


def process_folder(files, output_fp, pca_fp, models, classifier):
    #
    images = glob.glob(files + "/*", recursive=True)
    #
    print_message("Extracting features")
    features = classifier.extract_features(images, batch_size=64)
    #
    print_message("Generating predictions")
    predictions = classifier.classify(images, batch_size=64)
    #
    print_message("Applying PCA")
    pca = pickle.load(open(pca_fp, "rb"))
    pca, pca_features = features_apply_pca(features, pca=pca)
    print(
        "Applied %d pca_features of %d dimensions"
        % (len(pca_features), len(pca_features[0]))
    )
    #

    images = [img.replace(DATA_PATH, "") for img in images]

    pickle.dump([images, pca_features, predictions], open(output_fp, "wb"))
    print(
        "saved %d images, %d pca features, %d predictions"
        % (len(images), pca_features.shape[0], len(predictions))
    )


def main(args):

    # Init models and classifier
    models = Models()
    classifier = ImageClassifier(models.model, models.feat_extractor)

    # Empty dirs
    for f in glob.glob(images_start_folder + "/*"):
        os.remove(f)
    for f in glob.glob(images_end_folder + "/*"):
        os.remove(f)

    # Copy images
    copy_tree(args.start, images_start_folder)
    copy_tree(args.end, images_end_folder)

    # Classify Start Images
    print_message(
        "Classifying images from %s" % args.start, type="important", line="new"
    )
    process_folder(images_start_folder, start_features_fp, pca_fp, models, classifier)

    # Classify End Images
    print_message("Classifying images from %s" % args.end, type="important", line="new")
    process_folder(images_end_folder, end_features_fp, pca_fp, models, classifier)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="TBC")
    parser.add_argument(
        "-s",
        "--start",
        type=str,
        required=True,
        help="Source folder containing images for starting the path",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=str,
        required=True,
        help="Source folder containing images for ending the path",
    )
    args = parser.parse_args()
    main(args)

