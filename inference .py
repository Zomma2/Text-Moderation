###################################################################
##################### IMPORTS Pt.1 ###########################
###################################################################

# Import to list directories 
import os
### import for time and time monitoring 
import time
## Imports for Metrics operations 
import numpy as np 
## Imports for dataframes and .csv managment
import pandas as pd 
# TensorFlow library  Imports
import tensorflow as tf
import json
# Keras (backended with Tensorflow) Imports
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import CSVLogger

# Sklearn package for machine learining models 
## WILL BE USED TO SPLIT  TRAIN_VAL DATASETS
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Garbage Collectors
import gc
import sys


# Import Transformers to get Tokenizer for bert and bert models

import transformers
# from transformers import TFAutoModel, AutoTokenizer
# from transformers import RobertaTokenizer, TFRobertaModel
from transformers import DistilBertTokenizer, TFDistilBertModel , RobertaTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors


def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_masks=False, 
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    return np.array(enc_di['input_ids'])


def build_model(transformer, max_len=512):
    """
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def get_train_data():
    train = pd.read_csv('/content/jigsaw_mjy_train_val_openaug_523200.csv')
    #test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    return train





def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    data = data.read().decode("utf-8")[1:-1]
    MAX_LEN = 192
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    D = np.array([data])
    enc_di = tokenizer.batch_encode_plus(D,
                                         return_attention_masks=False, 
                                         return_token_type_ids=False,
                                         pad_to_max_length=True,
                                         max_length=MAX_LEN
                                        )
    
    x_test = np.array(enc_di['input_ids']) 
    return json.dumps({"instances": x_test.tolist()})


def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))

    response_content_type = context.accept_header
    prediction = data.content
    return prediction,response_content_type