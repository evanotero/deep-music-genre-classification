from common import load_track, get_layer_output_function
import numpy as np
from keras.layers import Input
from keras.models import model_from_yaml, Model
from keras import backend as K

class GenreRecognizer():

    def __init__(self, weighted_model):
        self.pred_fun = get_layer_output_function(weighted_model, 'output_realtime')
        print('Loaded model.')
    
    def recognize(self, track_path):
        print('Loading song', track_path)
        (features, duration) = load_track(track_path)
        features = np.reshape(features, (1,) + features.shape)
        return (self.pred_fun(features), duration)
