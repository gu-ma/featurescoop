# Testing with kmbs api
from tqdm import tqdm
import random
import utils
import requests

PORT1 = 5002
PORT2 = 5003

info = requests.get(f"http://0.0.0.0:{PORT1}/info?name=vitale_all_vgg16").json()
imgs = requests.get(f"http://0.0.0.0:{PORT1}/index?name=vitale_all_vgg16").json()

for i in tqdm(range(30)):

    idx1 = random.randint(0, len(imgs))
    idx2 = random.randint(0, len(imgs))
    path_idxs = requests.get(
        f"http://0.0.0.0:{PORT1}/path?name=vitale_all_vgg16&idx1={idx1}&idx2={idx2}"
    ).json()
    path = [imgs[idx] for idx in path_idxs]
    path = ",".join(path)
    results = requests.get(
        f"http://0.0.0.0:{PORT2}/getscenes?name=vitale_all&imgs={path}"
    ).json()
    print(results)
    # utils.video_make_sequence(results['scenes'], results['videos'], f"test{i}.mp4")

