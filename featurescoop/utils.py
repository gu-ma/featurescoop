import os
import bz2
import glob
import pickle
import shutil
import tempfile
import itertools
import subprocess
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from termcolor import colored
from collections import namedtuple
from moviepy.editor import VideoFileClip
from scenedetect import VideoManager, SceneManager
from scenedetect.scene_manager import (
    write_scene_list_html,
    generate_images,
)
from scenedetect.detectors import ContentDetector

# def system_clear_unused_folders
# Clear unused thumbs, scenes, features, etc...

# def system_generate_missing_files
# rebuild all feat, ec... missing


# def get_scene_info(images, id=0, max_duration=0, frames_per_scene=0):
#     """
#     Take an image ID as input
#     Return start and end time in second and a list of images id
#     """
#     # TODO: Extract a scene based on time

#     # Get all the info needed
#     image, video, folder, filename, time = get_info(images, id)
#     scenes_fp = Path(folder, SCENES_FN)

#     # if the scenes file does not exist yet, create it
#     if not os.path.isfile(scenes_fp):
#         scenes = video_extract_scenes(video, folder)
#         save_compressed_pickle(scenes, scenes_fp)
#     else:
#         scenes = load_compressed_pickle(scenes_fp)

#     # start and end time of the scene
#     start_time = [s for s in scenes if s <= time][-1]
#     end_time = [s for i, s in enumerate(scenes) if s > time or i == len(scenes) - 1][0]
#     if end_time - start_time > max_duration and max_duration != 0:
#         end_time = start_time + max_duration

#     # start and end id of the images
#     start_id = id
#     end_id = id
#     #
#     while True:
#         if start_id == 0:
#             break
#         if get_info(images, start_id - 1)[4] <= start_time:
#             break
#         if (get_info(images, start_id)[4] - get_info(images, start_id - 1)[4]) < 0:
#             break
#         start_id -= 1
#     #
#     while True:
#         if end_id >= len(images) - 1:
#             break
#         if get_info(images, end_id + 1)[4] > end_time:
#             break
#         if (get_info(images, end_id + 1)[4] - get_info(images, end_id)[4]) < 0:
#             break
#         end_id += 1
#     #
#     if frames_per_scene == 0 or end_id == start_id:
#         ids_list = list(range(start_id, end_id + 1))
#     else:
#         x = {
#             int(np.interp(i, [start_id, end_id], [0, frames_per_scene])): i
#             for i in range(start_id, end_id)
#         }
#         ids_list = list(x.values())
#     #
#     return start_time, end_time, ids_list, start_id, end_id


def get_all_scenes_duration(dir):
    """
    Get duration of all scenes saved in the dir
    """
    scenes_fps = glob.glob(f"{dir}**/scenes.pbz2")
    durations = []
    for scenes_fp in scenes_fps:
        scenes = load_compressed_pickle(scenes_fp)
        for scene in scenes:
            duration = scene[1].get_seconds() - scene[0].get_seconds()
            durations.append(duration)
    return durations


def get_info(img_fp, id=0):
    """
    Get info from a filename
    Returns: Path to video, video filename, scene number and scenes list
    """
    # .../source/foo/bar/mov1-Scene-004-01.jpg
    if os.path.isfile(img_fp):
        path_parts = Path(img_fp).parts
        folder = Path(*path_parts[:-1])
        video_fn = path_parts[-2]
        video_fp = Path(folder, video_fn + ".mp4")
        scene_num = int(path_parts[-1].split("-")[-2])
        scenes = load_compressed_pickle(os.path.join(folder, "scenes.pbz2"))
        Info = namedtuple("Info", ["video_fp", "video_fn", "scene_num", "scenes"])
        results = Info(video_fp, video_fn, scene_num, scenes)
    else:
        results = None

    return results


def video_convert_video(src_fp, dst_fp):
    """
    Convert video from the src_fp to a standard format and save it in dst_fp
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        dst_fn = Path(dst_fp).resolve().stem
        converted_vid_fp = os.path.join(tmpdirname, dst_fn + ".mp4")
        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            src_fp,
            "-b:a",
            "128k",
            "-b:v",
            "2M",
            "-c:a",
            "copy",
            "-c:v",
            "libx264",
            converted_vid_fp,
        ]
        subprocess.check_output(args)
        shutil.copy(converted_vid_fp, dst_fp)


def video_make_sequence(scenes, videos_fp, dst_fp):
    """
    Create video from a list of scenes / videos 
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        cut_videos = []
        # Cut
        for scene, video_fp in zip(scenes, videos_fp):
            start_time = scene[0]
            duration = scene[1] - scene[0]
            dst_fn = f"{Path(video_fp).resolve().stem}_{start_time}.ts" 
            cut_vid_fp = os.path.join(tmpdirname, dst_fn)
            print(start_time, duration)
            args = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                video_fp,
                "-ss",
                str(start_time),
                "-t",
                str(duration),
                "-vf",
                "scale=640:480",
                "-c:v",
                "libx264",
                "-crf",
                "1",
                "-c:a",
                "copy",
                cut_vid_fp
            ]
            subprocess.check_output(args)
            cut_videos.append(cut_vid_fp)
        # Concat
        print(cut_videos)
        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            f"concat:{'|'.join(cut_videos)}",
            "-c",
            "copy",
            dst_fp
        ]
        subprocess.check_output(args)


# def video_extract_frames(video, imgdir, fps=1):
#     """
#     Extract image frames from a video
#     Returns a list of file paths
#     """
#     # ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1
#     args = [
#         "ffprobe",
#         "-hide_banner",
#         "-loglevel",
#         "error",
#         "-show_entries",
#         "format=duration",
#         "-of",
#         "default=noprint_wrappers=1:nokey=1",
#         video,
#     ]
#     duration = float(subprocess.check_output(args))
#     print(
#         "Duration %.2f sec, fps %.2f, extracting %d frames"
#         % (duration, fps, (duration * fps + 1))
#     )

#     # There is a bug with the last second sometime, we just cut the video capture before for now
#     # https://trac.ffmpeg.org/wiki/Scaling
#     # args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-t", str(math.floor(duration)), "-i", video, "-vsync", "drop", "-vf", "fps="+str(fps), "-q:v", "10", os.path.join(imgdir, '%05d.jpg')]
#     args = [
#         "ffmpeg",
#         "-hide_banner",
#         "-loglevel",
#         "error",
#         "-t",
#         str(math.floor(duration)),
#         "-i",
#         video,
#         "-vsync",
#         "drop",
#         "-vf",
#         "fps="
#         + str(fps)
#         + ",scale=320:240:force_original_aspect_ratio=decrease,pad=320:240:(ow-iw)/2:(oh-ih)/2",
#         "-q:v",
#         "10",
#         os.path.join(imgdir, "%05d.jpg"),
#     ]
#     # args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-t", str(math.floor(duration)), "-i", video, "-vsync", "drop", "-vf", "scale=320:240:force_original_aspect_ratio=increase,crop=320:240:fps="+str(fps), "-q:v", "10", os.path.join(imgdir, '%05d.jpg')]
#     subprocess.call(args)
#     # rename
#     imgs = []
#     for p in sorted(Path(imgdir).glob("*.jpg")):
#         t = "%.2f" % ((float(p.name[:-4]) - 1) / fps)
#         new_filename = str(t) + ".jpg"
#         imgpath = os.path.join(p.parent, new_filename)
#         p.rename(imgpath)
#         imgs.append(imgpath)

#     return imgs


# def video_extract_scenes(video, threshold=0.3):
#     """
#     Extract scenes from a video and store them in a file
#     """
#     args = [
#         "ffprobe",
#         "-hide_banner",
#         "-show_frames",
#         "-loglevel",
#         "error",
#         "-of",
#         "csv",
#         "-f",
#         "lavfi",
#         "movie=" + video + ",select='gt(scene," + str(threshold) + ")'",
#     ]
#     output = subprocess.check_output(args, universal_newlines=True)
#     scenes = [float(line.split(",")[5]) for line in output.split("\n")[:-1]]
#     # Add zero and duration to the scenes
#     clip = VideoFileClip(video)
#     duration = float(clip.duration)
#     scenes = [0] + scenes + [duration]
#     print(
#         "%d Scenes extracted, average scene length %.2f s"
#         % (len(scenes) - 1, duration / float(len(scenes) - 1))
#     )
#     return scenes


def video_extract_scenes(
    video_path, output_dir, threshold=30.0, min_scene_len=15, frames_per_scene=3
):

    # Create video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    content_detector = ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
    scene_manager.add_detector(content_detector)

    # Base timestamp at frame 0 (required to obtain the scene list).
    base_timecode = video_manager.get_base_timecode()

    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()

    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Get cut and scene list
    cut_list = scene_manager.get_cut_list(base_timecode)
    scene_list = scene_manager.get_scene_list(base_timecode)

    video_name = os.path.basename(video_path).split(".")[0]

    # Save images
    generate_images(
        scene_list,
        video_manager,
        video_name,
        output_dir=output_dir,
        num_images=frames_per_scene,
        show_progress=True,
    )

    # Save an html summary
    images = glob.glob(f"{output_dir}/*")
    images.sort()
    images = [os.path.relpath(image, output_dir) for image in images]
    key_func = lambda x: "-".join(os.path.split(x)[-1].split("-")[:-1])
    image_filenames = [list(image) for _, image in itertools.groupby(images, key_func)]
    output_html_filename = os.path.join(output_dir, f"{video_name}_preview.html")
    write_scene_list_html(
        output_html_filename,
        scene_list,
        image_filenames=image_filenames,
        image_width=320,
        image_height=240,
    )

    return cut_list, scene_list


# def concatenate_scenes(images, frames_per_scene=0):
#     """
#     Returns a dictionary of scenes images with their start and stop time and the total number of scenes extracted
#     """
#     # TODO: Make this better
#     scenes = {}
#     i = 0
#     count = 0
#     pbar = tqdm(total=len(images))
#     #
#     while i < len(images):
#         scene = get_scene_info(images, i, frames_per_scene=frames_per_scene)
#         # return start_time, end_time, ids_list, start_id, end_id
#         # print(scene[:2])
#         # print('scene %03d: %.2f --> %.2f ' % (count, scene[0], scene[1]))
#         # print(scene[2])
#         # TODO: Use filename as dict key?
#         scenes.update({x: [scene[0], scene[1]] for x in scene[2]})
#         i = scene[4] + 1
#         count += 1
#         pbar.update(10)
#     #
#     pbar.close()
#     return scenes, count


# def concatenate_lists(files):
#     """
#     Returns a concatenated list
#     """
#     # TODO: Make it in batch?
#     all_lists = []
#     for file in tqdm(files):
#         all_lists += pickle.load(open(file, "rb"))
#     return all_lists


# def concatenate_arrays(files):
#     """
#     Return a concatenated array
#     """
#     # TODO: Make it in batch ?
#     length = pickle.load(open(files[0], "rb"))[0].shape[0]
#     all_arrays = np.empty(shape=[0, length], dtype=float)
#     for file in tqdm(files):
#         all_arrays = np.vstack((all_arrays, pickle.load(open(file, "rb"))))
#     return all_arrays


def print_message(msg, type="info", line=""):
    if type == "info":
        msg = colored("[INFO] " + msg, "cyan", attrs=[])
    if type == "warning":
        msg = colored("[WARNING] " + msg, "red", attrs=["bold", "underline"])
    if type == "important":
        msg = colored(msg, "white", "on_blue", attrs=["bold"])
    if type == "processing":
        msg = colored("[PROCESSING] " + msg, "yellow", attrs=[])
    if line == "new":
        msg = "----------\n" + msg
    print(msg)


def get_concatenated_images(images, indexes, thumb_height):
    #
    thumbs = []
    for id in indexes:
        img = Image.open(images[id])
        img = img.convert("RGB")
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    #
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image


def save_compressed_pickle(data, fn, incremental=False):
    with bz2.BZ2File(fn, "a" if incremental else "w") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_compressed_pickle(fn, incremental=False):
    data = []
    with bz2.BZ2File(fn, "rb") as f:
        if not incremental:
            data = pickle.load(f)
        else:
            try:
                while True:
                    data = [*data, *pickle.load(f)]
            except EOFError:
                pass
    return data
