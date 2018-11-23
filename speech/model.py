import numpy as np
import os

from keras.layers import Input, LSTM, RepeatVector, Dense, Activation
from keras.models import Model
from keras import metrics

class LSTM_AE():
	def __init__(self, n_frame, n_feat, n_dim, optimizer='Adadelta'):
		inputs = Input(shape=(n_frame, n_dim), name='input')
		enc_layer1 = LSTM(n_feat, return_sequences=False, name='encoder')(inputs)
		#enc_layer2 = LSTM(n_feat, return_sequences=False, name='encoder')(enc_layer1)
		temp = RepeatVector(n_frame)(enc_layer1)
		dec_layer3 = LSTM(n_dim, return_sequences=True)(temp)
		self.AE = Model(inputs, dec_layer3)
		self.encoder = Model(self.AE.inputs, self.AE.get_layer("encoder").output)
        
	def train(self, sequence, val_split=0.2, epochs=60):
		self.AE.compile(optimizer='adam', loss='mse')
		history = self.AE.fit(sequence, sequence, validation_split=val_split, nb_epochs=epochs)
		self.encoder = Model(self.AE.inputs, self.AE.get_layer("encoder").output)
		return history

	def feature(self, sequence):
		feat = self.encoder.predict(sequence, verbose=0)
		return feat


class LSTM_dim():
	def __init__(self, n_frame, n_feat, n_dim, optimizer='Adadelta'):
		inputs = Input(shape=(n_frame, n_dim), name='input')
		layer1 = LSTM(512, return_sequences=True)(inputs)
		layer2 = LSTM(256, return_sequences=False)(layer1)
		layer3 = Dense(n_feat, name='representation')(layer2)
		layer4 = Activation('relu')(layer3)
		layer5 = Dense(3)(layer4)
		layer6 = Activation('relu')(layer5)
		self.dim_LSTM = Model(inputs, layer6)
		self.rep = Model(self.dim_LSTM.inputs, self.dim_LSTM.get_layer('representation').output)

	def train(self, in_seq, out_seq, val_split=0.2, epochs=100):
		self.dim_LSTM.compile(optimizer='adam', loss='mse')
		history = self.dim_LSTM.fit(in_seq, out_seq, validation_split=val_split, nb_epochs=epochs)
		self.rep = Model(self.dim_LSTM.inputs, self.dim_LSTM.get_layer('representation').output)
		return history

	def feature(self, in_seq):
		return self.rep.predict(in_seq, verbose=0)


class LSTM_cat():
	def __init__(self, n_frame, n_feat, n_dim, optimizer='Adadelta'):
		inputs = Input(shape=(n_frame, n_dim), name='input')
		layer1 = LSTM(512, return_sequences=True)(inputs)
		layer2 = LSTM(256, return_sequences=False)(layer1)
		layer3 = Dense(n_feat, name='representation')(layer2)
		layer4 = Activation('relu')(layer3)
		layer5 = Dense(4)(layer4)
		layer6 = Activation('softmax')(layer5)
		self.cat_LSTM = Model(inputs, layer6)
		self.rep = Model(self.cat_LSTM.inputs, self.cat_LSTM.get_layer('representation').output)

	def train(self, in_seq, out_seq, val_split=0.2, epochs=30):
		self.cat_LSTM.compile(optimizer='adam', loss='categorical_crossentropy')
		history = self.cat_LSTM.fit(in_seq, out_seq, validation_split=val_split, nb_epochs=epochs, shuffle=True)
		self.rep = Model(self.cat_LSTM.inputs, self.cat_LSTM.get_layer('representation').output)
		return history

	def feature(self, in_seq):
		return self.rep.predict(in_seq, verbose=0)
