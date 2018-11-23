# Speech Emotion Recognition and Feature Extraction in IEMOCAP

**This repository contains**
* [Feature Extraction](https://github.com/makhilbabu/multi-modal-emotion-recognition/blob/master/speech/feature_extraction.py)
  * [Low-Level (Frame level)](https://github.com/makhilbabu/multi-modal-emotion-recognition/blob/master/speech/features.py)
    * Mel-Frequency Cepstral Coefficients (MFCC)
    * Linear Prediction Coefficients (LPC)
    * Zero Crossing Rate (ZCR)
    * Chroma spectrogram features
    * Residual signal
    * Residual Mel-Frequency Cepstral Coefficients (RMFCC)
  * High-Level (Utterance level)
    * Autoencoder (AE)
    * DNN-categorical-labels (cat)
    * DNN-dimensional-labels (dim)
    * Voice Quality (Jitter, Shimmer, HNR stats, U/V ratio) - but computed along low-level features

* [Models](https://github.com/makhilbabu/multi-modal-emotion-recognition/blob/master/speech/model.py)
  * Keras-based autoencoder for speech data
  * Keras-based DNN feature extraction using categorical labels

* [Feature-fusion](https://github.com/makhilbabu/multi-modal-emotion-recognition/blob/master/speech/classifiers.ipynb)
  * Neural Network Embedding
