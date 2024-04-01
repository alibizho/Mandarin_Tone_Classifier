from flask import Flask, render_template, jsonify
import os
import librosa
import pandas as pd
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
import random

app = Flask(__name__)
json_file = open('../model/model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)


# load weights into new model
# The model is by Taylor Burke
model.load_weights("../model/model.h5")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
print("Model Loaded")

file = open('../model/model_spec.txt', 'r') 
print(file.read())
file.close()

def decode(recording_name):
    audio, sampling_rate = librosa.load(recording_name, duration=1.5)
    
    user_mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=80) 
    padded_user_mfcc = np.asarray([pad_sequences(user_mfcc, padding='post', maxlen = 60)])
    user_audio_test_data = np.transpose(padded_user_mfcc, (0, 2, 1))
    prediction_accuracies = model.predict(user_audio_test_data)
    print(prediction_accuracies)
    # quick fix because the model is biased towards third tone
    if prediction_accuracies[0][2] < 0.6:
        second_largest = np.argsort(prediction_accuracies)[0][-2]
        predicted_tone = second_largest
    else:
        predicted_tone = np.where((prediction_accuracies > 0.5).astype("int32")[0] == 1)[0][0]

    tone = predicted_tone+ 1 
   
    return tone



@app.route("/")
def home():
    df = pd.read_csv('characters.csv')
    presented_character = df.iloc[random.randint(0, len(df) - 1)]['Simplified Character']
    actual_tone = df[df['Simplified Character'] == presented_character]['Tone'].values[0]
    pinyin = df[df['Simplified Character'] == presented_character]['Pinyin'].values[0]

    return render_template("home.html", character=presented_character, actual_tone=actual_tone, pinyin=pinyin)


@app.route("/record")   
def record():
    frame_rate = 44100
    duration = 1.4
    recording = sd.rec(int(duration * frame_rate), samplerate = frame_rate, channels = 1)
    recording_num = len(os.listdir('recordings/')) + 1
    recording_name = 'recordings/sample_audio_{}.mp3'.format(recording_num)
    sd.wait()
    write(recording_name, frame_rate, recording)
    tone = decode(recording_name)

    return jsonify(tone=str(tone))



if __name__ == "__main__":
    app.run(port=8000, debug=True)