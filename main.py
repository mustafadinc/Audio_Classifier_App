import sys, os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, \
    QHBoxLayout, QVBoxLayout, QLabel
from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtGui import QPixmap, QPalette
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import random
import os
import librosa
import librosa.display
import numpy as np
import math
import json
import tensorflow
from tensorflow import keras

MY_FILE = "EXAMPLE.wav"
PERSON_CLASSIFIER_MODEL_PATH = "person_classifier.h5"
GENDER_CLASSIFIER_MODEL_PATH = "gender_classifier.h5"

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.window_width, self.window_height = 1080, 800
        self.setMinimumSize(self.window_width, self.window_height)

        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        # AUDIO AND GRAPH PANEL ON THE RIGHT
        self.audio_panel = QVBoxLayout()
        #self.setLayout(self.layout)

        btn = QPushButton('Play', clicked=self.playAudioFile)
        self.audio_panel.addWidget(btn)

        volumeControl = QHBoxLayout()
        self.audio_panel.addLayout(volumeControl)

        btnVolumeUp = QPushButton('+', clicked=self.volumeUp)
        btnVolumeDown = QPushButton('-', clicked=self.volumeDown)
        butVolumeMute = QPushButton('Mute', clicked=self.volumeMute)
        volumeControl.addWidget(btnVolumeUp)
        volumeControl.addWidget(butVolumeMute)
        volumeControl.addWidget(btnVolumeDown)

        self.player = QMediaPlayer()

        self.plot_btn = QPushButton('Plot', clicked=self.spectrum)
        self.audio_panel.addWidget(self.plot_btn)

        self.figure,self.ax = plt.subplots(nrows=3, ncols=1, sharex=True)
        self.canvas = FigureCanvas(self.figure)
        self.audio_panel.addWidget(self.canvas)

        self.layout.addLayout(self.audio_panel)

        # INFORMATION PANEL ON THE RIGHT
        self.person_name_of_the_sound = QLabel(self)
        self.person_name_of_the_sound.setText("<font color=yellow>... Person Name ...</font>")
        self.person_name_of_the_sound.setAutoFillBackground(True)
        palette = QPalette()
        palette.setColor(QPalette.Window, Qt.blue)
        self.person_name_of_the_sound.setPalette(palette)
        self.person_name_of_the_sound.setAlignment(Qt.AlignCenter)

        self.person_gender_of_the_sound = QLabel(self)
        self.person_gender_of_the_sound.setText("<font color=yellow>... Person Gender ...</font>")
        self.person_gender_of_the_sound.setAutoFillBackground(True)
        palette1 = QPalette()
        palette1.setColor(QPalette.Window, Qt.darkBlue)
        self.person_gender_of_the_sound.setPalette(palette1)
        self.person_gender_of_the_sound.setAlignment(Qt.AlignCenter)

        information_panel = QVBoxLayout()
        information_panel.addWidget(self.person_name_of_the_sound)
        information_panel.addWidget(self.person_gender_of_the_sound)

        self.layout.addLayout(information_panel)

        self.showMaximized()

    def spectrum(self):
        ax = self.ax
        data, sample_rate = librosa.load(MY_FILE, sr=22050)
        # STFT -> spectrogram
        hop_length = 512  # fourier transforma girecek verilerin bölüm bölüm taranırken sağa tarafa doğru ne kadarlık bir kayma olacağı
        n_fft = 2048  # bir fourier transforma girecek veri sayısı

        # calculate duration hop length and window in seconds
        hop_length_duration = float(hop_length) / sample_rate
        n_fft_duration = float(n_fft) / sample_rate

        print("STFT hop length duration is: {}s".format(hop_length_duration))
        print("STFT window duration is: {}s".format(n_fft_duration))

        stft = librosa.stft(data, n_fft=n_fft, hop_length=hop_length)
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)

        mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, hop_length=hop_length)

        mfcc_mesh = librosa.display.specshow(mfcc,x_axis="time",ax=ax[0])

        mesh = librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hop_length, x_axis="time",
                                        y_axis="linear", ax=ax[1])

        librosa.display.waveplot(y=data, sr=sample_rate,ax=ax[2])

        ax[0].set(title='Mel-frequency cepstral coefficients (MFCCs)')
        ax[0].label_outer()

        ax[1].set(title='Log-frequency power spectrogram')
        ax[1].label_outer()

        ax[2].set(title='Waveshow (Time/Decibel)')
        ax[2].label_outer()

        self.figure.colorbar(mfcc_mesh, ax=self.ax, format="%+2.f dB")
        plt.title("Spectrogram")
        self.canvas.draw()
        self.figure.clf()
        #plt.show()

    """
    def plot(self):
        # random data
        data = [random.random() for i in range(10)]

        # clearing old figure
        self.figure.clear()

        # create an axis
        ax = self.figure.add_subplot(111)

        # plot data
        ax.plot(data, '*-')

        # refresh canvas
        self.canvas.draw()
    """

    def volumeUp(self):
        currentVolume = self.player.volume()  #
        print(currentVolume)
        self.player.setVolume(currentVolume + 5)

    def volumeDown(self):
        currentVolume = self.player.volume()  #
        print(currentVolume)
        self.player.setVolume(currentVolume - 5)

    def volumeMute(self):
        self.player.setMuted(not self.player.isMuted())

    def playAudioFile(self):
        full_file_path = os.path.join(os.getcwd(), MY_FILE)
        url = QUrl.fromLocalFile(full_file_path)
        content = QMediaContent(url)

        self.player.setMedia(content)
        self.player.play()
        self.getPersonName()
        self.getPersonGender()


    def getPersonName(self):
        print("Change person name")
        mfcc_list = self.getMFCCFromAudio()

        X = np.array(mfcc_list, dtype=object)
        X = np.asarray(X).astype('float32')
        np.squeeze(X).shape
        X = X[..., np.newaxis]  # array shape (1, number of sample that has number of mfcc in them, number of mfcc, 1)
        person_select_model = keras.models.load_model(PERSON_CLASSIFIER_MODEL_PATH)
        prediction = person_select_model.predict(X)
        predicted_index = np.argmax(prediction, axis=1)
        i = int(predicted_index)

        path = r"data_voices.json"
        data_info_dict = ""
        with open(path, "r") as fp:
            data = json.load(fp)
        people = data["mapping"]
        person = people[i]
        print(person)
        self.person_name_of_the_sound.setText("<font color=yellow>... {NAME} is spoken ...</font>".format(NAME=person))

    def getMFCCFromAudio(self):

        signal, librosa_sample_rate = librosa.load(MY_FILE)
        sample_rate = librosa_sample_rate # 22050
        hop_length = 512
        n_fft = 512
        #hop_length_duration = float(hop_length) / sample_rate
        #n_fft_duration = float(n_fft) / sample_rate

        num_segments = 1
        num_mfcc = 20
        track_duration = 3  # measured in seconds
        samples_per_track = sample_rate * track_duration
        samples_per_segment = int(samples_per_track / num_segments)
        num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

        mfcc_list = []
        for d in range(num_segments):

            # calculate start and finish sample for current segment
            start = samples_per_segment * d
            finish = start + samples_per_segment

            # extract mfcc
            mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                        hop_length=hop_length)
            mfcc = mfcc.T

            # store only mfcc feature with expected number of vectors
            if len(mfcc) == num_mfcc_vectors_per_segment:
                mfcc_list.append(mfcc.tolist())

        return mfcc_list

    def getPersonGender(self):
        gender_select_model = keras.models.load_model(GENDER_CLASSIFIER_MODEL_PATH)
        features = self.getMelSpecFromAudio()
        probability = gender_select_model.predict(features)
        male_prob = probability[0][0]
        female_prob = 1 - male_prob
        gender = ""
        if male_prob > female_prob:
            gender = "male"
        else:
            gender = "female"
        print(gender)

        self.person_gender_of_the_sound.setText("<font color=yellow>... {GENDER} ...</font>".format(GENDER=gender))

    def getMelSpecFromAudio(self):
        signal, sample_rate = librosa.load(MY_FILE)
        mel = np.mean(librosa.feature.melspectrogram(y=signal, sr=sample_rate).T, axis=0)
        result = np.array([])
        result = np.hstack((result, mel)) # stack arrays in column wise
        result = result.reshape(1, -1)
        return result


if __name__ == '__main__':
    # don't auto scale when drag app to a different monitor.
    # QApplication.setAttribute(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setStyleSheet('''
        QWidget {
            font-size: 20px;
        }
    ''')

    myApp = MyApp()
    myApp.show()

    try:
        sys.exit(app.exec_())
    except SystemExit:
        print('Closing Window...')


