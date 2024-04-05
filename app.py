import streamlit as st
import librosa
import soundfile
import os
import numpy as np
import pickle

# Load the MLP model from the pickle file
def load_model():
    with open("mlp_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Load the MLP model
model = load_model()

# Define function to extract features
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel_spec = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_spec))
    return result

# Define function to predict emotion
def predict_emotion(file):
    # Extract features from the audio file
    features = extract_feature(file)

    # Perform prediction using your model
    prediction = model.predict(features.reshape(1, -1))[0]

    return prediction

# Streamlit UI
st.set_page_config(page_title="Speech Emotion Recognition", page_icon=":microphone:")
st.title("Speech Emotion Recognition")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file (WAV format)", type=["wav"])

if uploaded_file is not None:
    # Save the uploaded file
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Display the uploaded file with some styling
    st.audio("temp.wav", format="audio/wav")

    # Perform prediction
    prediction = predict_emotion("temp.wav")

    # Display the predicted emotion with some styling
    st.subheader("Predicted Emotion")
    st.write(prediction)
    st.success("🎉 Prediction successful!")
