import pandas as pd
import numpy as np
import os
import csv

from sklearn.model_selection import StratifiedKFold

import spacy
import nltk
import string
import re
from unidecode import unidecode
from emoji import UNICODE_EMOJI

from time import time
from tqdm import tqdm


# Defining portuguese nltk resources
nltk.download('stopwords')
nltk.download('punkt')
portuguese_stopwords = nltk.corpus.stopwords.words('portuguese')
rslps_stemmer = nltk.stem.RSLPStemmer()
snowball_stemmer = nltk.stem.snowball.SnowballStemmer('portuguese', ignore_stopwords=False)

# Loading pt-br spacy model
nlp = spacy.load('pt_core_news_lg', disable=['tagger','parser','ner','entity_ruler','entity_linker','textcat'])

# Auxiliar preprocess function
def _preprocess_text(doc, remove_emojis=False):
    doc = re.sub(r"(http|www)\S+", "", doc) #remove URL
    doc = re.sub(r"\d{1,}.?\d{0,}", " 0 ", doc) #normalize numbers to zero
    doc = re.sub(r"[\⛤\¿\—\…\’\•\¡\°\º\´\!\"\#\%\&\\\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\\\]\^\_\`\{\|\}\~]{1,}", " ", doc) # remove special characters
    doc = re.sub(r"(?i)r? ?\${1,}", " $ ", doc) #normalize $

    # Replace (or not) emojis with description
    for emoji_i in UNICODE_EMOJI.keys():
        if remove_emojis==False:
            doc = doc.replace(emoji_i, ' ' + UNICODE_EMOJI[emoji_i].replace(':','') + ' ')
        else:
            doc = doc.replace(emoji_i, ' ')

    return doc

# Add folds columns to dataset 
def preprocess_dataset(input_file='B2W-Reviews01.csv', output_file='b2w_preprocessed.csv', nfolds=5, preprocess_corpus=True, text_column='review_text'):
    
    # Import raw dataframe
    try:
        data = pd.read_csv(os.path.join('0_data','datasets',input_file))
    except:
        data = pd.read_csv(os.path.join('0_data','datasets',input_file), sep=';')

    # Shuffle dataframe
    data = data.sample(random_state=0, frac=1)#.reset_index(drop=True)

    # Add folds columns related to target with 2 classes
    data['kfold_2'] = -1
    y2 = data['recommend_to_a_friend'].dropna().values
    kf2 = StratifiedKFold(n_splits=nfolds)
    for fold_, (_, test_idx) in enumerate(kf2.split(X=data.dropna(subset=['recommend_to_a_friend']), y=y2)):
        new_test_idx = data.dropna(subset=['recommend_to_a_friend']).index.values[test_idx]
        data.loc[new_test_idx, 'kfold_2'] = fold_

    # Add folds columns related to target with 5 classes
    data['kfold_5'] = -1
    y5 = data['overall_rating'].dropna().values
    kf5 = StratifiedKFold(n_splits=nfolds)
    for fold_, (_, test_idx) in enumerate(kf5.split(X=data.dropna(subset=['overall_rating']), y=y5)):
        new_test_idx = data.dropna(subset=['overall_rating']).index.values[test_idx]
        data.loc[new_test_idx, 'kfold_5'] = fold_

    # Preprocess corpus
    if preprocess_corpus==True:
        tqdm.pandas()
        data['preprocessed_w_emoji_' + text_column] =  data[text_column].progress_apply(lambda x: _preprocess_text(x, remove_emojis=False)) 
        data['preprocessed_wo_emoji_' + text_column] = data[text_column].progress_apply(lambda x: _preprocess_text(x, remove_emojis=True))
    
    # Save csv
    data.to_csv(os.path.join('0_data','datasets',output_file), index=False)


# Tokenize corpus
def tokenize_corpus(input_file, text_column, output_file):
    
    # Import raw dataframe
    try:
        data = pd.read_csv(os.path.join('0_data','datasets',input_file))
    except:
        data = pd.read_csv(os.path.join('0_data','datasets',input_file), sep=';')

    # Corpus
    corpus = data[text_column].astype(str)
    
    # Tokenize corpus through spacy
    tokenized_corpus_lemma = []
    tokenized_corpus_stem = []
    tokenized_corpus_none = []
    for doc in tqdm(nlp.pipe(corpus)):
        tokenized_doc_lemma = []
        tokenized_doc_stem = []
        tokenized_doc_none = []
        for token in doc:
            if (token.text not in portuguese_stopwords):

                # Lemmatization
                token_lemma = token.lemma_
                token_lemma = unidecode(token_lemma.lower()).strip()
                if token_lemma not in ('','[?]'):
                    tokenized_doc_lemma.append(token_lemma)

                # Stemming
                token_stem = rslps_stemmer.stem(token.text).lower()
                token_stem = unidecode(token_stem.lower()).strip()
                if token_stem not in ('','[?]'):
                    tokenized_doc_stem.append(token_stem)

                # Without normalization
                token_none = token.text
                token_none = unidecode(token_none.lower()).strip()
                if token_none not in ('','[?]'):
                    tokenized_doc_none.append(token_none)

        tokenized_corpus_lemma.append(tokenized_doc_lemma)
        tokenized_corpus_stem.append(tokenized_doc_stem)
        tokenized_corpus_none.append(tokenized_doc_none)

    # Saving tokenized_corpus to csv
    with open(os.path.join('0_data','datasets',output_file + '_lemma' + '.csv'),'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerows(tokenized_corpus_lemma)
    with open(os.path.join('0_data','datasets',output_file + '_stem' + '.csv'),'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerows(tokenized_corpus_stem)
    with open(os.path.join('0_data','datasets',output_file + '_none' + '.csv'),'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerows(tokenized_corpus_none)
                

if __name__ == '__main__':
    preprocess_dataset()
    tokenize_corpus(input_file='b2w_preprocessed.csv', text_column='preprocessed_w_emoji_review_text',  output_file='b2w_tokenized_w_emoji')
    tokenize_corpus(input_file='b2w_preprocessed.csv', text_column='preprocessed_wo_emoji_review_text', output_file='b2w_tokenized_wo_emoji')
