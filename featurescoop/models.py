import keras
from keras.models import Model


class Models(object):
    def __init__(self, weights="imagenet"):
        self.model = keras.applications.VGG16(weights=weights, include_top=True)
        # self.model.summary()
        self.feat_extractor = Model(
            inputs=self.model.input, outputs=self.model.get_layer("fc2").output
        )
        # self.feat_extractor.summary()

