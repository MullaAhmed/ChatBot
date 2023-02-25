import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from utils import *

############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    pass

def generate_response(input_text, model, vectorizer, corpus_texts):
    input_text = preprocess(input_text)
    input_vector = vectorizer.transform([input_text]).todense()
    X = np.zeros((1, 1, input_vector.shape[1]))
    X[0, 0, :] = input_vector
    
    y_pred = model.predict(X)
 
    best_match_index = np.argmax(y_pred)
    
    if (y_pred[0][best_match_index])>0.20:
      best_match = corpus_texts[best_match_index]
    else:
      best_match="Im not sure"

    return best_match


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    import json
    f = open('corpus.json')
    data = json.load(f)
    corpus_texts = data['facts']
    
    vectorizer, X = train_vectorizer(corpus_texts)
    y = np.eye(len(corpus_texts))

    input_dim = X.shape[1]
    output_dim = len(corpus_texts)

    X_train = np.zeros((len(corpus_texts), 1, input_dim))
    for i, text in enumerate(corpus_texts):
        X_train[i, 0, :] = vectorizer.transform([text]).todense()

    model = build_model(input_dim, output_dim)
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)
    model.fit(X_train, y, epochs=100, callbacks=[es])
    

    for text in request.text:
        
        response = generate_response(text, model, vectorizer, corpus_texts)
    
        output.append(response)

    return SimpleText(dict(text=output))
