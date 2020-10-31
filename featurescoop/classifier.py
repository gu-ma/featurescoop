import numpy as np
import tqdm

from keras.preprocessing import image

# specific to model
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions


class ImageClassifier(object):
    def __init__(self, model, feat_extractor):
        self.model = model
        self.feat_extractor = feat_extractor
        outputs = self.model.get_layer("fc2").output

    def prepare_image(self, filename, array_shape):
    """ 
    Preprocess an image and turn it into a np array 
    Returns an handle to the image itself, and a numpy array of the pixels to input the network
    """
        img = image.load_img(filename, target_size=array_shape)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        return img, x

    def extract_features(self, images, batch_size=32):
    """
    Returns a list of images and an array of features
    """
        sample_count = len(images)
        x = np.zeros((sample_count,) + self.feat_extractor.input_shape[1:4])

        print("Preparing images")
        for i, img in enumerate(images):
            x[i] = self.prepare_image(img, self.feat_extractor.input_shape[1:3])[1]

        print("Predicting features")
        features = self.feat_extractor.predict(x, batch_size=batch_size, verbose=1)

        return features

    def classify(self, images, batch_size=32):
    """
    Returns a list of images and an array of predictions (size 5)
    """
        sample_count = len(images)
        x = np.zeros((sample_count,) + self.model.input_shape[1:4])

        print("Preparing images")
        for i, img in enumerate(images):
            x[i] = self.prepare_image(img, self.model.input_shape[1:3])[1]

        print("Predicting labels")
        predictions = self.model.predict(x, batch_size=batch_size, verbose=1)
        predictions = decode_predictions(predictions)

        return predictions

