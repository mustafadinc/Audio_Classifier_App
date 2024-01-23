from PyQt5.QtWidgets import QMainWindow, QApplication , QWidget , QPushButton, QLabel, QFileDialog
        
from PyQt5 import uic , QtWidgets
import sys
from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import librosa
import librosa.display
import numpy as np
from PyQt5.QtGui import QPixmap, QPalette
from PyQt5.QtCore import Qt
import json
import math
import tensorflow
from tensorflow import keras
import os
import joblib
from sklearn.preprocessing import StandardScaler


PERSON_CLASSIFIER_MODEL_PATH = "person_classifier_02.h5"
GENDER_CLASSIFIER_MODEL_PATH = "gender_classifier.h5"
AGE_MODEL_PATH = "AgeModel.sav"

class myapp(QMainWindow):
    
    def __init__(self):
        
        super(myapp,self).__init__()
        self.MY_FILE = "" 
        
        
        
        #Load the ui file
        uic.loadUi("test.ui",self)
        
        self.show()
        #define our widgets
        
        self.layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.layout)
        self.button = self.findChild(QPushButton,"pushButton")
        self.buttonply = self.findChild(QPushButton,"playbutton")
        self.buttonpausee = self.findChild(QPushButton,"pausebutton")
        self.buttonstopp = self.findChild(QPushButton,"stopbutton")
        self.buttonagee = self.findChild(QPushButton,"agebutton")
        self.buttonperson = self.findChild(QPushButton,"personbutton")
        self.buttongender = self.findChild(QPushButton,"genderbutton")
        self.buttonplott = self.findChild(QPushButton,"plotbutton")
        
        self.widgett = self.findChild(QWidget,"widget")
        self.label = self.findChild(QLabel , "label")
        self.label2 = self.findChild(QLabel , "label_2")
        self.label3 = self.findChild(QLabel , "label_3")
        self.label4 = self.findChild(QLabel , "label_4")
        self.label5 = self.findChild(QLabel , "label_5")
       
        
        
        self.closebutton1= self.findChild(QPushButton , "closebutton")
        
        
        #click the dropdown box
        self.button.clicked.connect(self.buttonclick)
        self.buttonply.clicked.connect(self.play)
        self.buttonpausee.clicked.connect(self.pause)
        self.buttonstopp.clicked.connect(self.stop)
        self.buttonplott.clicked.connect(self.spectrum)
        self.buttongender.clicked.connect(self.getPersonGender)
        self.buttonperson.clicked.connect(self.getPersonName)
        self.buttonagee.clicked.connect(self.age_predict)
        self.closebutton1.clicked.connect(self.clear_all)
        
        
        
        
        
        
        self.player = QMediaPlayer(self)
        self.audio_panel = QtWidgets.QVBoxLayout(self.widgett)
        
        self.figure,self.ax = plt.subplots(nrows=3, ncols=1, sharex=True)
        self.canvas = FigureCanvas(self.figure)
        
        self.audio_panel.addWidget(self.canvas)
        
        
        # INFORMATION PANEL ON THE RIGHT
        
        
        
    def buttonclick(self):
        #self.label.setText("you clicked the button")
        fname =QFileDialog.getOpenFileName(self , "Open File" , "","All Files (*);;waw Files(*.wav)")
        
        
        
        if len(fname[0])>0:
            self.MY_FILE = fname[0]
            self.label.setText(self.MY_FILE)
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.MY_FILE)))
            if self.player.state()==QMediaPlayer.PlayingState:
                self.player.stop()
                
            
            self.setPlayingMode(False)
            
     
    def setPlayingMode(self,mode):
        if mode:
            self.playbutton.setEnabled(False)
            self.pausebutton.setEnabled(True)
            self.stopbutton.setEnabled(True)
        else:
            self.playbutton.setEnabled(True)
            self.pausebutton.setEnabled(False)
            self.stopbutton.setEnabled(False)
            
    
    def play(self):
        self.player.play()
        self.setPlayingMode(True)
        
        
    def pause(self):
        self.player.pause()
        self.setPlayingMode(False)
        
    def stop(self):
        self.player.stop()
        self.setPlayingMode(False)
    
    def spectrum(self):
        
        self.canvas.setParent(None)
        self.figure,self.ax = plt.subplots(nrows=3, ncols=1, sharex=True)
        self.canvas = FigureCanvas(self.figure)
        
        self.audio_panel.addWidget(self.canvas)
        ax = self.ax
        data, sample_rate = librosa.load(self.MY_FILE, sr=22050)
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

        librosa.display.waveshow(y=data, sr=sample_rate,ax=ax[2])

        ax[0].set(title='Mel-frequency cepstral coefficients (MFCCs)' )
        ax[0].label_outer()
        
        ax[1].set(title='Log-frequency power spectrogram')
        ax[1].label_outer()

        ax[2].set(title='Waveshow (Time/Decibel)')
        ax[2].label_outer()

        self.figure.colorbar(mfcc_mesh, ax=self.ax, format="%+2.f dB")
        #plt.title("Spectrogram")
        self.canvas.draw()
        #self.figure.clf()
        #self.canvas.clear()
        plt.show()
    

    
    def getPersonGender(self):
        gender_select_model = keras.models.load_model(GENDER_CLASSIFIER_MODEL_PATH)
        features = self.getMelSpecFromAudio()
        probability = gender_select_model.predict(features)
        male_prob = probability[0][0]
        female_prob = 1 - male_prob
        gender = ""
        if male_prob > female_prob:
            gender = "Erkek"
            
        else:
            gender = "Kadin"
            
        print(gender)
        self.label3.setText("<font color=red>... {GENDER} ...</font>".format(GENDER=gender))
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
        percentage = prediction[0][np.argmax(prediction)] * 100

        path = r"data_voices.json"
        data_info_dict = ""
        with open(path, "r") as fp:
            data = json.load(fp)
        
        people = data["mapping"]
        person = people[i]
        
        print(person)
        self.label2.setText("<font color=red>Konusan Kisi : {NAME} %{PERCENTAGE}</font>".format(NAME=person,PERCENTAGE=round(percentage,2)))
        self.pixmap=QPixmap(person)
        self.label4.setPixmap(self.pixmap)
        
    def getMFCCFromAudio(self):

        signal, librosa_sample_rate = librosa.load(self.MY_FILE)
        sample_rate = librosa_sample_rate # 22050
        hop_length = 512
        n_fft = 512
        #hop_length_duration = float(hop_length) / sample_rate
        #n_fft_duration = float(n_fft) / sample_rate

        num_segments = 1
        num_mfcc = 13
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
    def getMelSpecFromAudio(self):
        signal, sample_rate = librosa.load(self.MY_FILE)
        mel = np.mean(librosa.feature.melspectrogram(y=signal, sr=sample_rate).T, axis=0)
        result = np.array([])
        result = np.hstack((result, mel)) # stack arrays in column wise
        result = result.reshape(1, -1)
        return result
    
    def age_feature_extraction(self,path,sampling_rate=48000):

        features = list()
        
        audio, _ = librosa.load(path, sr=sampling_rate)
        gender = 1.0
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sampling_rate))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sampling_rate))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sampling_rate))
        features.append(gender)
        features.append(spectral_centroid)
        features.append(spectral_bandwidth)
        features.append(spectral_rolloff)
    
        mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate)
        for el in mfcc:
            features.append(np.mean(el))
    
        return features
    
    def age_predict(self):
        loaded_model = joblib.load("AgeModel.sav")
        ages = ["50ler","40lar","70ler","60lar","<20","30lar","20ler"]
        path = self.MY_FILE
        
        features=self.age_feature_extraction(path)
        features = np.array(features).reshape(1, -1)

        result = loaded_model.predict(features)
        age_value = [result[0]-1]
        
        print(ages[result[0]-1])
        
        self.label5.setText("<font color=red> YAS ARALIGI: {AGE} </font>".format(AGE=ages[result[0]-1]))
        
     
    def clear_all(self):
        self.label5.clear()
        self.label2.clear()
        self.label3.clear()
        self.label4.clear() 
        self.widgett.clear() #buraya bir daha bak
        
    
    #def predict_group(self):
     #   if self.genderr == "erkek":
      #      if self.agee< "30lar":
       #         self.label6.setText("You are young man")
        #    else:
         #       self.label6.setText("You are old man")
                
        
            
        
        
        
#Initialize the app

app = QApplication(sys.argv)
UIWindow=myapp()
app.exec_()


        


