import random
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping



# Function to preprocess text
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Function to tokenize and filter stop words
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    return filtered_tokens

# Function to train the vectorizer
def train_vectorizer(corpus):
    vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X

# Function to build the deep learning model
def build_model(input_dim, output_dim):
    model = Sequential()
    model.add(LSTM(512, input_shape=(None, input_dim), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(256))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

# # Train the vectorizer and deep learning model
# corpus_texts = corpus['facts']
# vectorizer, X = train_vectorizer(corpus_texts)
# y = np.eye(len(corpus_texts))

# input_dim = X.shape[1]
# output_dim = len(corpus_texts)

# X_train = np.zeros((len(corpus_texts), 1, input_dim))
# for i, text in enumerate(corpus_texts):
#     X_train[i, 0, :] = vectorizer.transform([text]).todense()

# model = build_model(input_dim, output_dim)
# es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)
# model.fit(X_train, y, epochs=100, callbacks=[es])

# # Save the weights file
# model.save_weights('deep_lstm_weights.h5')