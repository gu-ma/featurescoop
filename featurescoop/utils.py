import os
import bz2
import math
import pickle
import shutil
import tempfile
import subprocess
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from config import SCENES_FN
from igraph import Graph
from termcolor import colored
from scipy.spatial import distance
from moviepy.editor import VideoFileClip
from sklearn.decomposition import IncrementalPCA

# def system_clear_unused_folders
# Clear unused thumbs, scenes, features, etc...

# def system_generate_missing_files
# rebuild all feat, ec... missing


def get_scene_info(images, id=0, max_duration=0, frames_per_scene=0):
    """
    Take an image ID as input
    Return start and end time in second and a list of images id
    """
    # TODO: Extract a scene based on time

    # Get all the info needed
    image, video, folder, filename, time = get_info(images, id)
    scenes_fp = Path(folder, SCENES_FN)

    # if the scenes file does not exist yet, create it
    if not os.path.isfile(scenes_fp):
        scenes = video_extract_scenes(video, folder)
        save_compressed_pickle(scenes, scenes_fp)
    else:
        scenes = load_compressed_pickle(scenes_fp)

    # start and end time of the scene
    start_time = [s for s in scenes if s <= time][-1]
    end_time = [s for i, s in enumerate(scenes) if s > time or i == len(scenes) - 1][0]
    if end_time - start_time > max_duration and max_duration != 0:
        end_time = start_time + max_duration

    # start and end id of the images
    start_id = id
    end_id = id
    #
    while True:
        if start_id == 0:
            break
        if get_info(images, start_id - 1)[4] <= start_time:
            break
        if (get_info(images, start_id)[4] - get_info(images, start_id - 1)[4]) < 0:
            break
        start_id -= 1
    #
    while True:
        if end_id >= len(images) - 1:
            break
        if get_info(images, end_id + 1)[4] > end_time:
            break
        if (get_info(images, end_id + 1)[4] - get_info(images, end_id)[4]) < 0:
            break
        end_id += 1
    #
    if frames_per_scene == 0 or end_id == start_id:
        ids_list = list(range(start_id, end_id + 1))
    else:
        x = {
            int(np.interp(i, [start_id, end_id], [0, frames_per_scene])): i
            for i in range(start_id, end_id)
        }
        ids_list = list(x.values())
    #
    return start_time, end_time, ids_list, start_id, end_id


def get_info(images, id=0):
    """
    Get a filename (movie) from an id or path (thumb)
    return path to image, video and time in second
    """
    image = images[id]
    # .../source/foo/bar/0.00.jpg
    pathparts = Path(image).parts
    folder = Path(*pathparts[:-1])
    filename = pathparts[-2]
    video = Path(folder, filename + ".mp4")
    time = float(pathparts[-1][:-4])
    return image, video, folder, filename, time


def video_convert_video(src_fp, dst_fp):
    """
    Convert video from the src_fp to a standard format and save it in dst_fp
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        dst_fn = Path(dst_fp).resolve().stem
        converted_vid_fp = os.path.join(tmpdirname, dst_fn + ".mp4")
        # Switch to libx264
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


def video_extract_frames(video, imgdir, fps=1):
    """ 
    Extract image frames from a video
    Returns a list of file paths
    """
    # ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1
    args = [
        "ffprobe",
        "-hide_banner",
        "-loglevel",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video,
    ]
    duration = float(subprocess.check_output(args))
    print(
        "Duration %.2f sec, fps %.2f, extracting %d frames"
        % (duration, fps, (duration * fps + 1))
    )

    # There is a bug with the last second sometime, we just cut the video capture before for now
    # https://trac.ffmpeg.org/wiki/Scaling
    # args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-t", str(math.floor(duration)), "-i", video, "-vsync", "drop", "-vf", "fps="+str(fps), "-q:v", "10", os.path.join(imgdir, '%05d.jpg')]
    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-t",
        str(math.floor(duration)),
        "-i",
        video,
        "-vsync",
        "drop",
        "-vf",
        "fps="
        + str(fps)
        + ",scale=320:240:force_original_aspect_ratio=decrease,pad=320:240:(ow-iw)/2:(oh-ih)/2",
        "-q:v",
        "10",
        os.path.join(imgdir, "%05d.jpg"),
    ]
    # args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-t", str(math.floor(duration)), "-i", video, "-vsync", "drop", "-vf", "scale=320:240:force_original_aspect_ratio=increase,crop=320:240:fps="+str(fps), "-q:v", "10", os.path.join(imgdir, '%05d.jpg')]
    subprocess.call(args)
    # rename
    imgs = []
    for p in sorted(Path(imgdir).glob("*.jpg")):
        t = "%.2f" % ((float(p.name[:-4]) - 1) / fps)
        new_filename = str(t) + ".jpg"
        imgpath = os.path.join(p.parent, new_filename)
        p.rename(imgpath)
        imgs.append(imgpath)

    return imgs


def video_extract_scenes(video, threshold=0.3):
    """
    Extract scenes from a video and store them in a file
    """
    args = [
        "ffprobe",
        "-hide_banner",
        "-show_frames",
        "-loglevel",
        "error",
        "-of",
        "csv",
        "-f",
        "lavfi",
        "movie=" + video + ",select='gt(scene," + str(threshold) + ")'",
    ]
    output = subprocess.check_output(args, universal_newlines=True)
    scenes = [float(line.split(",")[5]) for line in output.split("\n")[:-1]]
    # Add zero and duration to the scenes
    clip = VideoFileClip(video)
    duration = float(clip.duration)
    scenes = [0] + scenes + [duration]
    print(
        "%d Scenes extracted, average scene length %.2f s"
        % (len(scenes) - 1, duration / float(len(scenes) - 1))
    )
    return scenes


def features_get_closest_image(features, id=0, feature=[], num_neighbors=5):
    """
    Return the closests image id
    """
    # TODO: replace with annoy, remove pca features from parameters
    # use a globally stored distance matrix
    feat_from = features[id] if id != 0 else feature
    distances = [distance.euclidean(feat_from, feat_to) for feat_to in features]
    id_closest = sorted(range(len(distances)), key=lambda k: distances[k])[
        1 : num_neighbors + 1
    ]
    return id_closest


def features_get_distance(features, id1, id2):
    """
    Return the distance between 2 images
    """
    # TODO: replace with annoy, remove pca features from parameters
    d = distance.euclidean(features[id1], features[id2])
    return d


def concatenate_scenes(images, frames_per_scene=0):
    """
    Returns a dictionary of scenes images with their start and stop time and the total number of scenes extracted
    """
    # TODO: Make this better
    scenes = {}
    i = 0
    count = 0
    pbar = tqdm(total=len(images))
    #
    while i < len(images):
        scene = get_scene_info(images, i, frames_per_scene=frames_per_scene)
        # return start_time, end_time, ids_list, start_id, end_id
        # print(scene[:2])
        # print('scene %03d: %.2f --> %.2f ' % (count, scene[0], scene[1]))
        # print(scene[2])
        # TODO: Use filename as dict key?
        scenes.update({x: [scene[0], scene[1]] for x in scene[2]})
        i = scene[4] + 1
        count += 1
        pbar.update(10)
    #
    pbar.close()
    return scenes, count


def concatenate_lists(files):
    """
    Returns a concatenated list
    """
    # TODO: Make it in batch?
    all_lists = []
    for file in tqdm(files):
        all_lists += pickle.load(open(file, "rb"))
    return all_lists


def concatenate_arrays(files):
    """
    Return a concatenated array 
    """
    # TODO: Make it in batch ?
    length = pickle.load(open(files[0], "rb"))[0].shape[0]
    all_arrays = np.empty(shape=[0, length], dtype=float)
    for file in tqdm(files):
        all_arrays = np.vstack((all_arrays, pickle.load(open(file, "rb"))))
    return all_arrays


def features_apply_pca(
    features, components_count=300, pca="", chunk_size=500, batch_size=32
):
    """
    reduce the number of features with PCA
    """
    # TODO: use explained_variance_ratio_ instead of components_count

    n = features.shape[0]
    if chunk_size < n:
        chunk_size = n

    if not pca:

        pca = IncrementalPCA(n_components=components_count, batch_size=batch_size)
        pca.fit(features)

        # if components_count>n or batch_size>components_count:
        #   print_message('number of components > number of features or batch size > number of components')
        #   return

        # for i in range(0, n//chunk_size):
        #   start = i*chunk_size
        #   end = (i+1)*chunk_size if (i!=n//chunk_size-1) else n
        #   pca.partial_fit(features[start:end])

    # pca_features = np.empty(shape=[n, components_count], dtype=float)

    # for i in range(0, n//chunk_size):
    #   start = i*chunk_size
    #   end = (i+1)*chunk_size if (i!=n//chunk_size-1) else n
    #   pca_features[start:end] = pca.transform(features[start:end])

    # pca = IncrementalPCA(n_components=components_count, batch_size=batch_size)

    # pca = PCA(.95)
    # Use partial_fit:
    # https://stackoverflow.com/questions/31428581/incremental-pca-on-big-data
    # https://stackoverflow.com/questions/44334950/how-to-use-sklearns-incrementalpca-partial-fit
    # https://stackoverflow.com/questions/32857029/python-scikit-learn-pca-explained-variance-ratio-cutoff/32857305#32857305
    # pca.fit(features)

    pca_features = pca.transform(features)
    return pca, pca_features


# TODO: Add options to use Graph-Tool, save distance matrix?
def graph_build_graph(features, kNN=30):

    features = np.array(features)
    feat_count = len(features)

    graph = Graph(directed=True)
    graph.add_vertices(feat_count)

    # matrix of cosine distances between features
    print("1) Calculating cosine distance matrix")
    dist_matrix = distance.cdist(features, features, "cosine").astype(np.float)

    edges = np.empty(shape=[0, 2], dtype=int)
    weights = np.empty(shape=[0, kNN], dtype=float)

    print("2) Sorting and assigning distances")
    # TODO: do that without loop?
    for i, dist in enumerate(tqdm(dist_matrix)):
        id1 = np.repeat(i, kNN).astype(np.int)
        id2 = np.arange(feat_count, dtype=int)
        # sort indexes
        sort_id = np.argsort(dist)[1 : kNN + 1]
        # reorder distances and indexs
        id2 = id2[sort_id][:kNN]
        dist = dist[sort_id][:kNN]
        # save edges and weights
        edges = np.append(edges, np.stack((id1, id2), axis=1), axis=0)
        weights = np.append(weights, dist)

    graph.add_edges(edges)
    graph.es["weight"] = weights

    return graph


def graph_get_shortest_paths(graph, id1, id2):
    """
    Return the shortest path between 2 ids
    """
    paths = graph.get_shortest_paths(
        id1, to=id2, mode=1, output="vpath", weights="weight"
    )
    for path in paths:
        if not "" in path:
            return path


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
