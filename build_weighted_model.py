from common import GENRES
import os
import h5py
import numpy as np

SEED = 42
N_LAYERS = 3
FILTER_LENGTH = 5
CONV_FILTER_COUNT = 256
LSTM_COUNT = 256
BATCH_SIZE = 32
EPOCH_COUNT = 80


class TorchCRNNModel:
    def __init__(self, weights_filepath):
        import torch
        import torch.nn as nn

        self.weights_filepath = weights_filepath

        with h5py.File(weights_filepath, 'r') as f:
            mw = f['model_weights'] if 'model_weights' in f else f
            c1_w = np.array(mw['convolution_1']['convolution_1']['kernel:0'])
            c1_b = np.array(mw['convolution_1']['convolution_1']['bias:0'])
            c2_w = np.array(mw['convolution_2']['convolution_2']['kernel:0'])
            c2_b = np.array(mw['convolution_2']['convolution_2']['bias:0'])
            c3_w = np.array(mw['convolution_3']['convolution_3']['kernel:0'])
            c3_b = np.array(mw['convolution_3']['convolution_3']['bias:0'])
            lstm_w = np.array(mw['lstm_1']['lstm_1']['kernel:0'])
            lstm_u = np.array(mw['lstm_1']['lstm_1']['recurrent_kernel:0'])
            lstm_b = np.array(mw['lstm_1']['lstm_1']['bias:0'])
            dense_w = np.array(mw['time_distributed_1']['time_distributed_1']['kernel:0'])
            dense_b = np.array(mw['time_distributed_1']['time_distributed_1']['bias:0'])

        self.conv1 = nn.Conv1d(128, CONV_FILTER_COUNT, kernel_size=FILTER_LENGTH)
        self.conv1.weight.data = torch.from_numpy(c1_w.transpose(2, 1, 0)).float()
        self.conv1.bias.data = torch.from_numpy(c1_b).float()

        self.conv2 = nn.Conv1d(CONV_FILTER_COUNT, CONV_FILTER_COUNT, kernel_size=FILTER_LENGTH)
        self.conv2.weight.data = torch.from_numpy(c2_w.transpose(2, 1, 0)).float()
        self.conv2.bias.data = torch.from_numpy(c2_b).float()

        self.conv3 = nn.Conv1d(CONV_FILTER_COUNT, CONV_FILTER_COUNT, kernel_size=FILTER_LENGTH)
        self.conv3.weight.data = torch.from_numpy(c3_w.transpose(2, 1, 0)).float()
        self.conv3.bias.data = torch.from_numpy(c3_b).float()

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lstm_w = torch.from_numpy(lstm_w).float()
        self.lstm_u = torch.from_numpy(lstm_u).float()
        self.lstm_b = torch.from_numpy(lstm_b).float()
        self.units = LSTM_COUNT

        self.dense = nn.Linear(LSTM_COUNT, len(GENRES))
        self.dense.weight.data = torch.from_numpy(dense_w.T).float()
        self.dense.bias.data = torch.from_numpy(dense_b).float()

    @staticmethod
    def hard_sigmoid(x):
        import torch
        return torch.clamp(0.2 * x + 0.5, 0.0, 1.0)

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            if x.ndim == 2:
                x = x.unsqueeze(0)
            # x: (batch, time, 128) -> (batch, 128, time)
            x = x.permute(0, 2, 1)

            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))

            # (batch, channels=256, downsampled_time) -> (batch, downsampled_time, 256)
            x = x.permute(0, 2, 1)

            wx_b = torch.matmul(x, self.lstm_w) + self.lstm_b
            batch_size, seq_len, _ = x.shape
            h = torch.zeros(batch_size, self.units, dtype=x.dtype, device=x.device)
            c = torch.zeros(batch_size, self.units, dtype=x.dtype, device=x.device)

            h_seq = []
            u = self.lstm_u
            for t in range(seq_len):
                z = wx_b[:, t, :] + torch.matmul(h, u)
                z0 = z[:, :self.units]
                z1 = z[:, self.units : 2 * self.units]
                z2 = z[:, 2 * self.units : 3 * self.units]
                z3 = z[:, 3 * self.units :]

                i = self.hard_sigmoid(z0)
                f = self.hard_sigmoid(z1)
                c = f * c + i * torch.tanh(z2)
                o = self.hard_sigmoid(z3)
                h = o * torch.tanh(c)
                h_seq.append(h.unsqueeze(1))

            if len(h_seq) > 0:
                lstm_out = torch.cat(h_seq, dim=1)
            else:
                lstm_out = torch.zeros(batch_size, 0, self.units)

            logits = self.dense(lstm_out)
            probs = F.softmax(logits, dim=-1)
            return probs.detach().cpu().numpy()

    def get_layer_output_function(self, layer_name):
        return lambda x: [self.predict(x)]


def build_weighted_model(weights_filepath):
    print('Building model...')
    try:
        from keras.models import Model
        from keras.layers import Input
        n_features = 128
        input_shape = (None, n_features)
        model_input = Input(input_shape, name='input')
        model = crnn(model_input)
        model.load_weights(weights_filepath)
        return model
    except Exception:
        # Fall back to native PyTorch inference engine
        return TorchCRNNModel(weights_filepath)


def crnn(model_input):
    from keras.models import Model
    from keras.optimizers import RMSprop
    from keras import backend as K
    from keras.layers import Dense, Lambda, Dropout, Activation, LSTM, \
        TimeDistributed, Conv1D, MaxPooling1D

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
