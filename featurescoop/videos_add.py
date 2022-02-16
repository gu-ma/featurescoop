# -*- coding: utf-8 -*-
import os
import re
import glob
import json
import shutil
import argparse
from pathlib import Path

from utils import (
    video_extract_scenes,
    video_convert_video,
    print_message,
    save_compressed_pickle,
)


def main(args):

    # Basic way of doing this. The settings file will be ovewritten each time
    json_fp = os.path.join('settings', f'{args.name}.json')
    if os.path.isfile(json_fp):
        answer = input(f"Setting file '{json_fp}' exists, overwrite? (y/n) : ")
        if answer == 'y':
            json.dump(vars(args), open(json_fp, 'w'), indent=4)
    else:
        json.dump(vars(args), open(json_fp, 'w'), indent=4)

    # exit()

    videos = glob.glob(args.input + "/**/*.*", recursive=True)
    videos.sort()

    for i, video in enumerate(videos):

        print_message(f"video {i+1} of {len(videos)} \n──{video}", type="processing")

        force = args.force

        # Create folders
        subpath = [p for p in Path(video).parts if p not in Path(args.input).parts]
        subfolders = subpath[:-1]
        filename = Path(video).resolve().stem
        filename = re.sub("[^A-Za-z0-9_]+", "-", filename)
        output_dir = Path(args.output, args.source, *subfolders, filename)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Working in directory {output_dir}")

        # Copy and convert video
        video_fp = os.path.join(output_dir, filename + ".mp4")
        if not os.path.isfile(video_fp) or force:
            if args.noconvert:
                print_message("Copying video")
                shutil.copy(video, video_fp)
            else:
                print_message("Converting video")
                video_convert_video(video, video_fp)

        # Extract Scenes
        scenes_fp = os.path.join(output_dir, "scenes.pbz2")
        cuts_fp = os.path.join(output_dir, "cuts.pbz2")
        if not os.path.isfile(scenes_fp) or force:
            print_message("Extracting scenes")
            cuts, scenes = video_extract_scenes(
                video_path=video,
                output_dir=output_dir,
                threshold=args.threshold,
                min_scene_len=args.min_scene_len,
                frames_per_scene=args.frames_per_scene,
            )
            save_compressed_pickle(scenes, scenes_fp)
            save_compressed_pickle(cuts, cuts_fp)
            # force = True

        # # Extract frames
        # frames_fp = os.path.join(output_dir, FRAMES_FN)
        # if not os.path.isfile(frames_fp) or force:
        #     for p in Path(output_dir).glob("*.jpg"):
        #         p.unlink()
        #     print_message(f"Extracting frames ({args.frames_per_second} frames_per_second)")
        #     frames = video_extract_frames(video, path, fps=args.frames_per_second)
        #     print_message(f"Concatenating {'all' if args.frames_per_scene==0 else args.fpc} frames per scene")
        #     scenes, count = concatenate_scenes(frames, frames_per_scene=args.frames_per_scene)
        #     frames = [frames[i] for i in scenes.keys()]
        #     # frames_relative = [img.replace(DATA_PATH, "") for img in frames]
        #     save_compressed_pickle(frames, frames_fp)
        #     print_message("Saved %d frames from %d scenes" % (len(frames), count))
        #     force = True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Add new videos")
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="Name of the dataset",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Source folder containing the videos to classify",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Folder where the results would be saved",
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
        "-msl",
        "--min_scene_len",
        type=int,
        help="Minimum scene length (in frames)",
        default=15,
    )
    parser.add_argument(
        "-fpsc",
        "--frames_per_scene",
        type=int,
        help="Number of frames to extract from each scene",
        default=1,
    )
    parser.add_argument(
        "-th",
        "--threshold",
        type=float,
        help="Thresold for scenes extraction, low threshold makes short scenes",
        default=30.00,
    )
    parser.add_argument(
        "-nc",
        "--noconvert",
        help="Don't convert all video files",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "-f",
        "--force",
        help="Force recreating ALL files",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "-fps",
        "--frames_per_second",
        type=float,
        help="Frame per sec if extracting frames from videos",
        default=1,
    )

    args = parser.parse_args()

    # Make path absolute
    args.input = str(Path(args.input).resolve())
    args.output = str(Path(args.output).resolve())

    main(args)
