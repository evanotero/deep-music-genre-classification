from build_weighted_model import build_weighted_model
from genre_recognizer import GenreRecognizer
from common import GENRES
import numpy as np
import os
import json
import uuid
from random import random
import time
import tornado
import tornado.ioloop
import tornado.web
import tornado.httpserver
import librosa
import soundfile as sf
from optparse import OptionParser

STATIC_PATH = os.path.join(os.path.dirname(__file__), 'static')
UPLOADS_PATH = os.path.join(os.path.dirname(__file__), 'uploads')

genre_recognizer = None


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render(os.path.join(STATIC_PATH, 'index.html'))


class PlayHandler(tornado.web.RequestHandler):
    def get(self):
        self.render(os.path.join(STATIC_PATH, 'play.html'))


class UploadHandler(tornado.web.RequestHandler):
    def post(self):
        file_info = self.request.files['filearg'][0]
        file_name = file_info['filename']
        file_extension = os.path.splitext(file_name)[1].lower()
        file_uuid = str(uuid.uuid4())

        if not os.path.exists(UPLOADS_PATH):
            os.makedirs(UPLOADS_PATH)

        uploaded_orig_path = os.path.join(UPLOADS_PATH, file_uuid + file_extension)
        uploaded_mp3_path = os.path.join(UPLOADS_PATH, file_uuid + '.mp3')

        with open(uploaded_orig_path, 'wb') as f:
            f.write(file_info['body'])

        # play.js expects an mp3 file at uploads/<uuid>.mp3
        if file_extension == '.mp3':
            analysis_path = uploaded_orig_path
        else:
            try:
                y, sr = librosa.load(uploaded_orig_path, mono=True)
                sf.write(uploaded_mp3_path, y, sr)
                analysis_path = uploaded_mp3_path
            except Exception as e:
                print('Conversion to MP3 failed, copying original:', e)
                with open(uploaded_mp3_path, 'wb') as f:
                    f.write(file_info['body'])
                analysis_path = uploaded_orig_path

        (predictions, duration) = genre_recognizer.recognize(analysis_path)
        genre_distributions = self.get_genre_distribution_over_time(predictions, duration)
        json_path = os.path.join(UPLOADS_PATH, file_uuid + '.json')
        with open(json_path, 'w') as f:
            f.write(json.dumps(genre_distributions))
        self.finish('"{}"'.format(file_uuid))

    def get_genre_distribution_over_time(self, predictions, duration):
        """
        Turns the matrix of predictions given by a model into a dict mapping
        time in the song to a music genre distribution.
        """
        preds = np.array(predictions)
        preds = np.reshape(preds, (-1, len(GENRES)))
        n_steps = preds.shape[0]
        if n_steps == 0:
            return []
        delta_t = duration / n_steps

        def get_genre_distribution(step):
            return {genre_name: float(preds[step, genre_index])
                    for (genre_index, genre_name) in enumerate(GENRES)}

        return [((step + 1) * delta_t, get_genre_distribution(step)) for step in range(n_steps)]


application = tornado.web.Application([
    (r'/', MainHandler),
    (r'/play', PlayHandler),
    (r'/play.html', PlayHandler),
    (r'/static/(.*)', tornado.web.StaticFileHandler, {
        'path': STATIC_PATH
    }),
    (r'/uploads/(.*)', tornado.web.StaticFileHandler, {
        'path': UPLOADS_PATH
    }),
    (r'/upload', UploadHandler),
], debug=True)

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-w', '--weights', dest='weights_path',
                      default=os.path.join(os.path.dirname(__file__),
                                           'models/weights.best.hdf5'),
                      help='load keras model WEIGHTS hdf5 file', metavar='WEIGHTS')
    parser.add_option('-p', '--port', dest='port',
                      default=8000,
                      help='run server at PORT', metavar='PORT')
    options, args = parser.parse_args()
    genre_recognizer = GenreRecognizer(build_weighted_model(options.weights_path))

    port = int(options.port)
    server = tornado.httpserver.HTTPServer(application, max_buffer_size=104857600)
    server.listen(port)
    print('Server running at http://localhost:{}/'.format(port))
    tornado.ioloop.IOLoop.current().start()
