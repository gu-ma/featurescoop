## Experimental repo

You probably don't want to fork it :stuck_out_tongue_winking_eye:

## Steps

1) Scrap videos using `youtube-dl` or `instaloader`. You can use `tools/scrapping.py` to _automate_ this step.
2) `videos_add.py`: Copy and convert videos into a folder in (see `config.py` for list of options) Then extract for each video:
    * Scenes - using `ffmpeg / ffprobe`
    * Frames - as images
    * Features for each frame - using the `classifier`
    * Predictions for each frame (labels) - using the `classifier`
3) `videos_process.py`: Builds (or load) single files concatenating all:
    * Scenes
    * Frames
    * Features (Reducing dimensions using PCA)
    * Predictions
It also create a file holding the graph connecting all features.
4) `videos_test.py`: Runs different test such as finding closest images and finding path between 2 frames
5) `startend_images_encode.py` is used to encode some images to start and end the animation. `startend_images_getpath.py` is used to test the start and end image (this is a quick last minute fix that needs to be removed)
6) `build_annoy_index.py` encode pregenerated lines of poetry using `spacy` and builds an index of those lines using `annoy`
7) `api.py` starts a local api which can be called to `getpath` between 2 images or `GetSimilarVideos`

## Requirements
P 3.6. Need to add version for all packages

### Python
* keras
* tensorflow
* igraph
* numpy
* scipy
* pickle
* matplotlib
* tqdm
* sklearn
* rasterfairy
* moviepy
* PIL ??
* imageio ??
* instaloader
* annoy
* spacy + en-core-web-sm / en_vectors_web_lg
* termcolor
* colorama

### Nonpython
* youtube-dl
* ffmpeg / ffprobe

## Vid description + text comparison

## TODO
import yt video etc... 
db?