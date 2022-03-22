import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import models, layers, optimizers
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import bz2
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import re

import os
print(os.listdir("../data/amazon_reviews"))
base = "../data/amazon_reviews"

def get_labels_and_texts(file):
    labels = []
    texts = []
    count = 0
    with open(base + '/train.ft.txt', "r") as a_file:
       
        for line in a_file:
            if count < 10000:
                #stripped_line = line.strip()
                #x = line.decode("utf-8")
                x = line
                labels.append(int(x[9]) - 1)
                texts.append(x[10:].strip())
                count = count+1
                
            else:
                return np.array(labels), texts


import re

def normalize_texts(texts):
    NON_ALPHANUM = re.compile(r'[\W]')
    NON_ASCII = re.compile(r'[^a-z0-1\s]')
    normalized_texts = []
    for text in texts:
        lower = text.lower()
        no_punctuation = NON_ALPHANUM.sub(r' ', lower)
        no_non_ascii = NON_ASCII.sub(r'', no_punctuation)
        normalized_texts.append(no_non_ascii)
    return normalized_texts