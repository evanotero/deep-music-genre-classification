# Deep Music Genre Classification

Using deep learning to categorize music as time progresses through spectrogram analysis.

## Authors
- Evan Otero
- Ribhi El-Zaru
- Nicholas Loeper
- Andrew Heimerman

## About
Our goal is to create a machine learning model capable of continuously predicting the probability distribution of a song's genre in real time as it plays. The project combines a Convolutional Recurrent Neural Network (CRNN: CNN + LSTM) with an interactive web visualizer (Tornado, Web Audio API, Canvas) to dynamically display how the network's belief of genre evolves throughout a track.

## Background
The model is inspired by and expands upon previous works in music information retrieval, notably:
- [Piotr Kozakowski and Bartosz Michalak, Convolutional-Recurrent Neural Network for Live Music Genre Recognition (DeepSound)](http://deepsound.io/music_genre_recognition.html)
- [Grzegorz Gwardys and Daniel Grzywczak, Deep Image Features in Music Information Retrieval](http://ijet.pl/index.php/ijet/article/view/10.2478-eletel-2014-0042/53)
- [Sander Dieleman, Recommending Music on Spotify with Deep Learning](http://benanne.github.io/2014/08/05/spotify-cnns.html)

While previous works (such as DeepSound) classified across 10 genres on the smaller GTZAN dataset (1,000 tracks), this project scales to the **Free Music Archive (FMA)** dataset, classifying over 16 diverse genres with real-time frame-by-frame distributions.

## Data

### Mel-Spectrogram Representation
Audio files are converted into 2-dimensional log mel-spectrograms using [Librosa](https://librosa.org/). Audio waveforms are transformed to spectral frequencies:
- **Window Size ($N_{\text{fft}}$)**: 2048 samples (~93 ms at 22,050 Hz)
- **Hop Length / Stride**: 1024 samples
- **Mel Filter Bins ($N_{\text{mels}}$)**: 128 bins
- **Log Scaling**: Logarithmic scaling maps sound power to decibels (dB), matching human hearing perception of loudness and pitch.

### Dataset
The model was trained on tracks from the [Free Music Archive (FMA)](https://github.com/mdeff/fma) dataset (Defferrard et al., 2017):
- Evaluated on tracks across 16 genres: *International, Blues, Jazz, Classical, Old-Time/Historic, Country, Pop, Rock, Easy Listening, Soul/R&B, Electronic, Folk, Spoken, Hip-Hop, Experimental, Instrumental*.
- Preprocessed from MP3s into structured serialized matrices, resulting in 21,695 songs (7.15 GB of log-mel features).
- Dataset split: **70% training**, **30% test/validation**.

## Model

### Architecture
The network combines 1D temporal convolutions for local feature extraction with an LSTM for long-range sequential structure:

| Layer | Type | Configuration | Output Shape |
|---|---|---|---|
| **Input** | Audio Features | Log mel-spectrogram slices | `(Batch, Time, 128)` |
| **Conv 1** | Conv1D + ReLU + MaxPool1D | 256 filters, length 5, stride 2 pooling | `(Batch, Time/2, 256)` |
| **Conv 2** | Conv1D + ReLU + MaxPool1D | 256 filters, length 5, stride 2 pooling | `(Batch, Time/4, 256)` |
| **Conv 3** | Conv1D + ReLU + MaxPool1D | 256 filters, length 5, stride 2 pooling | `(Batch, Time/8, 256)` |
| **Dropout** | Dropout | Rate 0.5 (training only) | `(Batch, Time/8, 256)` |
| **LSTM** | Recurrent Layer | 256 units, return sequences = True | `(Batch, Time/8, 256)` |
| **Dropout** | Dropout | Rate 0.5 (training only) | `(Batch, Time/8, 256)` |
| **Dense** | TimeDistributed Dense | 16 units, linear | `(Batch, Time/8, 16)` |
| **Real-time Output** | Softmax | Frame-level genre distribution | `(Batch, Time/8, 16)` |
| **Merged Output** | Temporal Mean Pooling | Overall song classification | `(Batch, 16)` |

- **Optimizer**: RMSprop ($\text{lr} = 10^{-5}$)
- **Loss**: Categorical Cross-Entropy (weighted by class frequency)

## Results

- **Validation Accuracy**: Reached **54.52%** accuracy on multi-class classification across 16 genre categories, significantly higher than random guessing (~6.25%).
- **Temporal Predictions**: The model captures genre transitions and shifts within songs (e.g., intros, solos, verses).
- **Genre Correlations**: The predicted probability distributions reveal intuitive relationships between genres (e.g., strong correlations between Classical & Jazz, Instrumental & Electronic, and Pop & Soul/R&B).

## Interactive Visualizer & Web App

The project includes an interactive web interface powered by Tornado:
- **Audio Upload**: Upload `.mp3`, `.wav`, or other audio files through an animated SiriWave upload interface.
- **Audio Transcoding**: Non-MP3 files are automatically converted and prepared for browser playback.
- **Dynamic Pie Chart**: Real-time distribution chart updating every 100 ms synchronized with playback via [Chart.js](http://www.chartjs.org/).
- **Particle System**: An audio-reactive canvas particle visualizer powered by the Web Audio API and [Sketch.js](https://brm.io/sketch.js/), with particles dynamically tinted to the colors of the dominant genres over time.

## Installation & Usage

### Prerequisites
- Python 3.9+ (compatible with modern Python 3.9–3.14 on macOS Apple Silicon and Linux)
- Pip

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/evanotero/deep-music-genre-classification.git
   cd deep-music-genre-classification
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Server
Start the local server using the launch script:
```bash
./run.sh
```

Or directly via python:
```bash
python server.py --port 8000
```

Open **http://localhost:8000/** in your browser:
1. Select and upload an audio file (`.mp3` or `.wav`).
2. The server analyzes the spectrogram and redirects to the playback visualizer.
3. Enjoy the real-time genre prediction and audio-reactive particle display.

## Slides & Report
- **Presentation Deck**: [Using Deep Learning to Categorize Music as Time Progresses Through Spectrogram Analysis](https://docs.google.com/presentation/d/1MANAML13S-PBGx8bsbI8gdaKGPJ2bPd_UNbA3E2txdg/edit?usp=sharing)
- **Full Report**: [Music_Genre_Recognition-3.pdf](Music_Genre_Recognition-3.pdf)
