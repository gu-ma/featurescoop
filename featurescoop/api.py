import os
import glob
import json

from flask import Flask, request
from flask_restful import Resource, Api
from flask import jsonify

import utils

app = Flask(__name__)
api = Api(app)


def load_settings(dir):

    settings = {}
    for fp in glob.glob(f"{dir}/*.*"):
        s = json.load(open(fp, "r"))
        settings[s["name"]] = s

    return settings


class Settings(Resource):
    def get(self):

        return jsonify(settings)


class GetScenes(Resource):
    def get(self):

        name = request.args.get("name")
        imgs = request.args.get("imgs")

        if name and imgs:

            img_fp = settings[name]["output"]
            imgs = imgs.split(",")
            scenes = []
            videos = []

            for img in imgs:
                img = os.path.join(img_fp, img)
                img_info = utils.get_info(img)

                if img_info:
                    scene = img_info.scenes[img_info.scene_num - 1]
                    # duration = float(scene[1] - scene[0])
                    scenes.append([scene[0].get_seconds(), scene[1].get_seconds()])
                    videos.append(str(img_info.video_fp))
                else:
                    scenes.append(None)
                    videos.append(None)

            results = {"scenes": scenes, "videos": videos}

        else:
            results = "Missing arguments"

        return jsonify(results)


api.add_resource(Settings, "/settings")
api.add_resource(GetScenes, "/getscenes")


if __name__ == "__main__":
    settings = load_settings("settings")
    app.config["JSON_SORT_KEYS"] = False
    app.run(host="0.0.0.0", port=5003)
