# -*- coding: utf-8 -*-
import os
import re
import glob
import shutil
import argparse
from pathlib import Path

from classifier import ImageClassifier
from models import Models
from config import DATA_PATH, SCENES_FN, FRAMES_FN, FEATURES_FN, PREDICTIONS_FN

from utils import video_extract_frames
from utils import video_extract_scenes
from utils import video_convert_video
from utils import concatenate_scenes
from utils import print_message
from utils import save_compressed_pickle, load_compressed_pickle


def main(args):

    print_message(
        "Classifying videos from %s" % args.input, type="important", line="new"
    )

    # Init models and classifier
    models = Models()
    classifier = ImageClassifier(models.model, models.feat_extractor)

    videos = glob.glob(args.input + "/**/*.*", recursive=True)
    videos.sort()

    for i, video in enumerate(videos):

        print_message(
            "Processing video %d of %d \n──%s" % (i + 1, len(videos), video),
            type="processing",
        )

        force = args.force

        # Create folder
        subpath = [
            p
            for p in Path(video).parts
            if p not in Path(args.input).parts
        ]
        subfolders = subpath[:-1]
        filename = Path(video).resolve().stem
        filename = re.sub("[^A-Za-z0-9_]+", "-", filename)
        path = Path(DATA_PATH, args.source, *subfolders, filename)
        os.makedirs(path, exist_ok=True)
        print("Working in directory %s" % path)

        # Copy and convert video
        video_fp = os.path.join(path, filename + ".mp4")
        if not os.path.isfile(video_fp) or force:
            if args.noconvert:
                print_message("Copying video")
                shutil.copy(video, video_fp)
            else:
                print_message("Converting video")
                video_convert_video(video, video_fp)

        # Extract Scenes
        scenes_fp = os.path.join(path, SCENES_FN)
        if not os.path.isfile(scenes_fp) or force:
            print_message("Extracting scenes")
            scenes = video_extract_scenes(video, args.threshold)
            save_compressed_pickle(scenes, scenes_fp)
            force = True
        else:
            scenes = load_compressed_pickle(scenes_fp)

        # Extract frames
        frames_fp = os.path.join(path, FRAMES_FN)
        if not os.path.isfile(frames_fp) or force:
            for p in Path(path).glob("*.jpg"):
                p.unlink()
            print_message("Extracting frames")
            frames = video_extract_frames(video, path, fps=args.fps)
            print_message(
                "Concatenating scenes with %d frames per scene (0 = all frames)"
                % args.fpsc
            )
            scenes, count = concatenate_scenes(frames, frames_per_scene=args.fpsc)
            frames = [frames[i] for i in scenes.keys()]
            # frames_relative = [img.replace(DATA_PATH, "") for img in frames]
            save_compressed_pickle(frames, frames_fp)
            print_message("Saved %d frames from %d scenes" % (len(frames), count))
            force = True
        else:
            frames = load_compressed_pickle(frames_fp)

        # Extract features
        features_fp = os.path.join(path, FEATURES_FN)
        if not os.path.isfile(features_fp) or force:
            print_message("Extracting features from frames")
            features = classifier.extract_features(frames)
            save_compressed_pickle(features, features_fp)
            print_message(
                "Saved %d features of size %d in a %s"
                % (features.shape[0], features.shape[1], type(features))
            )
            force = True
        else:
            features = load_compressed_pickle(features_fp)

        # Extract predictions
        predictions_fp = os.path.join(path, PREDICTIONS_FN)
        if not os.path.isfile(predictions_fp) or force:
            print_message("Generating predictions from frames")
            predictions = classifier.classify(frames)
            save_compressed_pickle(predictions, predictions_fp)
            print_message(
                "Saved %d predictions of size %d in a %s"
                % (len(predictions), len(predictions[0]), type(predictions))
            )
        else:
            predictions = load_compressed_pickle(predictions_fp)


if __name__ == "__main__":

    # TODO: use docopt
    parser = argparse.ArgumentParser(description="Add new videos")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Source folder containing the videos to classify",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        help="Source (e.g: youtube, vimeo, etc...",
        required=True,
    )
    parser.add_argument(
        "-fps",
        "--fps",
        type=float,
        help="Frame per sec when extracting frames from videos",
        default=1,
    )
    parser.add_argument(
        "-fpsc",
        "--fpsc",
        type=int,
        help="Number of frames to extract from each scene (0 = all)",
        default=0,
    )
    parser.add_argument(
        "-th",
        "--threshold",
        type=float,
        help="Thresold for scenes extraction, low threshold makes short scenes",
        default=0.1,
    )
    parser.add_argument(
        "-nc",
        "--noconvert",
        help="Don't convert all video files",
        action="store_true"
    )
    parser.add_argument(
        "-f",
        "--force",
        help="Force recreating ALL files",
        action="store_true"
    )

    args = parser.parse_args()
    main(args)
