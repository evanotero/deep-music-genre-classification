from common import GENRES
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import Model
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Dropout, Activation, LSTM, \
    TimeDistributed, Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import numpy as np
import _pickle as pickle
from optparse import OptionParser
from sys import stderr, argv
import os

SEED = 42
N_LAYERS = 3
FILTER_LENGTH = 5
CONV_FILTER_COUNT = 256
LSTM_COUNT = 256
BATCH_SIZE = 32
EPOCH_COUNT = 80


def crnn(model_input):
    layer = model_input
    for i in range(N_LAYERS):
        # Convolutional layer names are used by extract_filters.py
        layer = Conv1D(
            nb_filter=CONV_FILTER_COUNT,
            filter_length=FILTER_LENGTH,
            name='convolution_' + str(i + 1))(layer)
        layer = Activation('relu')(layer)
        layer = MaxPooling1D(2)(layer)

    layer = Dropout(0.5)(layer)
    layer = LSTM(LSTM_COUNT, return_sequences=True)(layer)
    layer = Dropout(0.5)(layer)
    layer = TimeDistributed(Dense(len(GENRES)))(layer)
    layer = Activation('softmax', name='output_realtime')(layer)
    time_distributed_merge_layer = Lambda(
        function=lambda x: K.mean(x, axis=1),
        output_shape=lambda shape: (shape[0],) + shape[2:],
        name='output_merged')
    model_output = time_distributed_merge_layer(layer)
    model = Model(model_input, model_output)
    opt = RMSprop(lr=0.00001)  # Optimizer
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])

    return model


def build_model(x_train):
    print('Building model...')

    n_features = x_train.shape[2]
    input_shape = (None, n_features)
    model_input = Input(input_shape, name='input')

    return crnn(model_input)


def train_model(data):
    # Inputs
    x = data['x']
    y = data['y']

    # Use 30% of the data for test validatino.
    (x_train, x_val, y_train, y_val) = train_test_split(x, y, test_size=0.3, random_state=SEED)

    # Compile the CRNN.
    model = build_model(x_train)

    # Calculate the class weight, since dataset is unbalanced.
    y_weights = class_weight.compute_sample_weight('balanced', y_train)

    # Create callbacks for training.
    tb_callback = TensorBoard(log_dir='./logs/4', histogram_freq=1, batch_size=32, write_graph=True, write_grads=False,
                              write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                              embeddings_metadata=None)
    checkpoint_callback = ModelCheckpoint('./models/weights.best.hdf5', monitor='val_acc', verbose=1,
                                          save_best_only=True, mode='max')
    callbacks_list = [tb_callback, checkpoint_callback]

    # Fit the model and get training history.
    print('Training...')
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH_COUNT,
                        validation_data=(x_val, y_val), verbose=1, sample_weight=y_weights, callbacks=callbacks_list)

    return model, history


def show_summary_stats(history):
    # List all data in history
    print(history.history.keys())

    # Summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-m', '--model_path', dest='model_path',
                      default=os.path.join(os.path.dirname(__file__),
                                           './models/model.yaml'),
                      help='path to the output model YAML file', metavar='MODEL_PATH')
    parser.add_option('-w', '--weights_path', dest='weights_path',
                      default=os.path.join(os.path.dirname(__file__),
                                           './models/weights.h5'),
                      help='path to the output model weights hdf5 file',
                      metavar='WEIGHTS_PATH')
    options, args = parser.parse_args()

    pickle_data_0 = pickle.load(open('../ai-data/data_part0.pkl', 'rb'))
    pickle_data_1 = pickle.load(open('../ai-data/data_part1.pkl', 'rb'))
    pickle_data_2 = pickle.load(open('../ai-data/data_part2.pkl', 'rb'))
    pickle_data_3 = pickle.load(open('../ai-data/data_part3.pkl', 'rb'))

    pickle_data_concat = {
        'x': np.concatenate((pickle_data_0['x'], pickle_data_1['x'], pickle_data_2['x'], pickle_data_3['x'])),
        'y': np.concatenate((pickle_data_0['y'], pickle_data_1['y'], pickle_data_2['y'], pickle_data_3['y']))}

    m, h = train_model(pickle_data_concat)

    with open(options.model_path, 'w') as f:
        f.write(m.to_yaml())

    m.save_weights(options.weights_path)

    plot_model(m, to_file='model.png')  # Save model graph.

    show_summary_stats(h)
