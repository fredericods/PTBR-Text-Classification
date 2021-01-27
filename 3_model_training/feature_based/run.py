import os
from os import listdir
from os.path import isfile, join

from feature_based import train_all_folds
from sklearn.linear_model import LogisticRegression

import itertools
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

# Define params to loop
list_nclasses = [2,5]
list_vectorizer_type = ['count','tfidf']
list_ngram_range = [(1,1)]
list_C = [0.10, 0.25, 0.50, 0.75, 1.00, 2.50, 5.00, 7.50, 10.0]
list_emoji = ['w','wo']
list_norm_type = ['stem','lemma','none']
list_vocab_size = {
    'stem':  [5e3, 10e3, 15e3, 20e3, None],
    'lemma': [5e3, 10e3, 15e3, 20e3, 25e3, 30e3, 35e3, None],
    'none':  [5e3, 10e3, 15e3, 20e3, 25e3, 30e3, 35e3, 40e3, 45e3, None]
}

stem_params =  list(itertools.product(list_nclasses, list_vectorizer_type, list_ngram_range, list_C, list_emoji, ['stem'],  list_vocab_size['stem']))
lemma_params = list(itertools.product(list_nclasses, list_vectorizer_type, list_ngram_range, list_C, list_emoji, ['lemma'], list_vocab_size['lemma']))
none_params =  list(itertools.product(list_nclasses, list_vectorizer_type, list_ngram_range, list_C, list_emoji, ['none'],  list_vocab_size['none']))

list_params = stem_params + lemma_params + none_params

def aux_train_validate_loop(params_i):
    # Get params from tuple
    nclasses_i, vectorizer_type_i, ngram_range_i, C_i, emoji_i, norm_type_i, vocab_size_i = params_i
    
    # Make vocab_size_i an integer
    if vocab_size_i != None:
        vocab_size_i = int(vocab_size_i)

    # Define corpus/input and dict/output filenames
    output_filename_i = f'reglog__{nclasses_i}__{vectorizer_type_i}__{C_i}__{emoji_i}__{norm_type_i}__{vocab_size_i}__{ngram_range_i[0]}_{ngram_range_i[1]}'
    corpus_file_i = f'b2w_tokenized_{emoji_i}_emoji_{norm_type_i}.csv'
    
    # Define classifier
    reglog_i = LogisticRegression(random_state=0, solver='saga', C=C_i)
    
    # Train and validate model
    train_all_folds(
        OUTPUT_FILENAME=output_filename_i,
        CORPUS_FILE=corpus_file_i,
        CLASSIFIER=reglog_i,
        NCLASSES=nclasses_i,
        VECTORIZER_TYPE=vectorizer_type_i,
        NGRAM_RANGE_=ngram_range_i,
        VOCAB_SIZE=vocab_size_i
        )

# Save dict results for each combination of params
#num_cores = multiprocessing.cpu_count()
#Parallel(n_jobs=num_cores)(delayed(aux_train_validate_loop)(i) for i in list_params[180:])

aux_train_validate_loop(list_params[181])