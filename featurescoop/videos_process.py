import os
import glob
import pickle
import argparse

from config import (
    DATA_PATH,
    SCENES_FN,
    FRAMES_FN,
    FEATURES_FN,
    PCA_FEATURES_FN,
    PREDICTIONS_FN,
    GRAPH_FN,
    PCA_FN,
)
from utils import print_message
from utils import features_apply_pca
from utils import graph_build_graph
from utils import concatenate_lists, concatenate_arrays, concatenate_scenes
from utils import save_compressed_pickle, load_compressed_pickle


# TODO: check for errors for all steps
def main(args):

    print_message(
        "Processing Features, Predictions, Scenes and graph",
        type="important",
        line="new",
    )

    # Build / Load frames
    all_frames_fp = os.path.join(DATA_PATH, FRAMES_FN)
    if not os.path.isfile(all_frames_fp) or args.force:
        files = glob.glob(DATA_PATH + "/**/" + FRAMES_FN, recursive=True)
        counter = 0
        print_message("Concatenating frames", line="new")
        for file in files:
            data = load_compressed_pickle(file)
            counter += len(data)
            save_compressed_pickle(data, all_frames_fp, incremental=True)
        print(f'Saved {counter} frames')
    print_message("Loading frames", line="new")
    frames = load_compressed_pickle(all_frames_fp, incremental=True)
    print(f'Loaded {len(frames)} frames')

    # frames = [DATA_PATH + img for img in frames]

    # Build / Load scenes
    all_scenes_fp = os.path.join(DATA_PATH, SCENES_FN)
    if not os.path.isfile(all_scenes_fp) or args.force:
        files = glob.glob(DATA_PATH + "/**/" + SCENES_FN, recursive=True)
        print_message("Concatenating scenes", line="new")
        for file in files:
            data = load_compressed_pickle(file)
            print(data)
            save_compressed_pickle(data, all_scenes_fp, incremental=True)
        print(f'Saved {counter} frames')

    else:
        print_message("Loading scenes", line="new")
        scenes = pickle.load(open(all_scenes_fp, "rb"))
        print("Loaded %d frames from %d scenes" % (len(scenes), count))

    exit()
    
    # Build / Load features
    all_features_fp = os.path.join(DATA_PATH, FEATURES_FN)
    if not os.path.isfile(all_features_fp) or args.force:
        files = glob.glob(DATA_PATH + "/**/" + FEATURES_FN, recursive=True)
        print_message("Concatenating features", line="new")
        features = concatenate_arrays(files)
        print("Processed %d features of %d dimensions" % features.shape[0:2])
        pickle.dump(features, open(all_features_fp, "wb"))
        print("Saved %d features of %d dimensions" % features.shape[0:2])
    else:
        print_message("Loading features", line="new")
        features = pickle.load(open(all_features_fp, "rb"))
        print("Loaded %d features of %d dimensions" % features.shape[0:2])

    # Apply PCA
    all_pca_features_fp = os.path.join(DATA_PATH, PCA_FEATURES_FN)
    pca_fp = os.path.join(DATA_PATH, PCA_FN)
    if not os.path.isfile(all_pca_features_fp) or args.force:
        # # Load start and end frames + features
        # img_start, features_start, predictions_start = pickle.load(open(start_features_fp, 'rb'))
        # img_end, features_end, predictions_end = pickle.load(open(end_features_fp, 'rb'))

        # # Add the start and end features to encode
        # features = np.vstack((features, features_start, features_end))

        print_message(
            "Applying PCA \nNumber of features: %d \nNumber of components: %d \nBatch size: %d \nChunk size: %d"
            % (features.shape[0], args.pcacount, args.batch_size, args.chunk_size)
        )

        pca, pca_features = features_apply_pca(
            features,
            components_count=args.pcacount,
            pca="",
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
        )

        pickle.dump(pca, open(pca_fp, "wb"))
        pickle.dump(pca_features, open(all_pca_features_fp, "wb"))
        #
        # pickle.dump([img_start, pca_features[len(frames):len(frames)+len(img_start)], predictions_start], open(start_features_fp, 'wb'))
        # pickle.dump([img_end, pca_features[-len(img_end):], predictions_end], open(end_features_fp, 'wb'))
        #
        # pca_features = pca_features[:len(frames)]

        print(
            "Saved PCA files and %d PCA features of %d dimensions"
            % pca_features.shape[0:2]
        )
    else:
        print_message("Loading pca features", line="new")
        pca_features = pickle.load(open(all_pca_features_fp, "rb"))
        print("Loaded %d pca features of %d dimensions" % pca_features.shape[0:2])

    # Build / Load predictions
    all_predictions_fp = os.path.join(DATA_PATH, PREDICTIONS_FN)
    files = glob.glob(DATA_PATH + "/**/" + PREDICTIONS_FN, recursive=True)
    if not os.path.isfile(all_predictions_fp) or args.force:
        print_message("Concatenating predictions", line="new")
        predictions = concatenate_lists(files)
        pickle.dump(predictions, open(all_predictions_fp, "wb"))
        print(
            "Saved %d predictions of %d dimensions"
            % (len(predictions), len(predictions[0]))
        )
    else:
        print_message("Loading predictions", line="new")
        predictions = pickle.load(open(all_predictions_fp, "rb"))
        print(
            "Loaded %d predictions of %d dimensions"
            % (len(predictions), len(predictions[0]))
        )

    # Build / load graph
    graph_fp = os.path.join(DATA_PATH, GRAPH_FN)
    if not os.path.isfile(graph_fp) or args.force:
        # Build and save graph
        print_message("Building graph with %d kNN" % args.knn, line="new")
        graph = graph_build_graph(pca_features, args.knn)
        pickle.dump(graph, open(graph_fp, "wb"))
        print(
            "Saved graph: %d edges, %d vertices, %d kNN"
            % (graph.ecount(), graph.vcount(), graph.ecount() / graph.vcount())
        )
    else:
        print_message("Loading graph", line="new")
        graph = pickle.load(open(graph_fp, "rb"))
        print(
            "Loaded graph: %d edges, %d vertices, %d kNN"
            % (graph.ecount(), graph.vcount(), graph.ecount() / graph.vcount())
        )


if __name__ == "__main__":

    # TODO: use docopt
    parser = argparse.ArgumentParser(
        description="Process all features and predictions files, reduce dimensions of features"
    )
    parser.add_argument(
        "-k", 
        "--knn",
        type=int,
        help="K Nearest Neighbours",
        default=15
    )
    parser.add_argument(
        "--pcacount",
        type=int,
        help="Number of component for each features to keep (for PCA)",
        default=300,
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        help="Chunk size (for PCA)",
        default=500
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size (for PCA)",
        default=32
    )
    parser.add_argument(
        "-f",
        "--force",
        help="Force recreating ALL files",
        action="store_true"
    )

    args = parser.parse_args()
    main(args)

