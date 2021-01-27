import pandas as pd
import numpy as np

from tqdm import tqdm
from time import time

import os
from os import listdir
from os.path import isfile, join
import gc
import joblib

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix


def train_fold(dataset, corpus, classifier, fold=0, nclasses=2, vectorizer_type='count', ngram_range_=(1,1), vocab_size=None):
    
    # Define target feature column and kfold column based on the number of classes
    target_feature = {2:'recommend_to_a_friend', 5:'overall_rating'}[nclasses]
    kfold_column = f'kfold_{nclasses}'
    
    # Drop instances with missing values on target feature
    dataset = dataset.dropna(subset=[target_feature])
    
    # Split corpus into train and validation
    corpus_train = list(map(corpus.__getitem__, dataset[dataset[kfold_column]!=fold].index.values))
    corpus_valid = list(map(corpus.__getitem__, dataset[dataset[kfold_column]==fold].index.values))
    
    # Define train and validation target features
    y_train = dataset[dataset[kfold_column]!=fold][target_feature].values
    y_valid = dataset[dataset[kfold_column]==fold][target_feature].values

    # Define vectorizer type: count or tf-idf
    if vectorizer_type=='count':
        vectorizer = CountVectorizer(tokenizer=lambda x:x, lowercase=False, ngram_range=ngram_range_, max_features=vocab_size, token_pattern=None)
    else:
        vectorizer = TfidfVectorizer(tokenizer=lambda x:x, lowercase=False, ngram_range=ngram_range_, max_features=vocab_size, token_pattern=None)
    
    # Fit vectorizer to training corpus
    # Transforming training and validation corpora into vectorizer shape
    start_time_vectorizer = time()
    vectorizer.fit(corpus_train)
    X_train = vectorizer.transform(corpus_train)
    X_valid = vectorizer.transform(corpus_valid)
    end_time_vectorizer = time()
    
    # Fit model and make predictions
    start_time_model = time()
    classifier.fit(X_train, y_train)
    y_train_pred = classifier.predict(X_train)
    y_train_pred_proba = classifier.predict_proba(X_train)
    y_valid_pred = classifier.predict(X_valid)
    y_valid_pred_proba = classifier.predict_proba(X_valid)
    end_time_model = time()
    
    # Model metrics
    if nclasses > 2:
        dict_metrics = {
            'vectorizer_time': end_time_vectorizer - start_time_vectorizer,
            'model_time': end_time_model - start_time_model,
            
            'vocab_size': len(vectorizer.vocabulary_.keys()),
            
            'accuracy_train': accuracy_score(y_train, y_train_pred),
            'accuracy_valid': accuracy_score(y_valid, y_valid_pred),
            
            'log_loss_train': log_loss(y_train, y_train_pred_proba, labels=[1,2,3,4,5]),
            'log_loss_valid': log_loss(y_valid, y_valid_pred_proba, labels=[1,2,3,4,5]),

            'macro_f1_train': f1_score(y_train, y_train_pred, average='macro'),
            'macro_f1_valid': f1_score(y_valid, y_valid_pred, average='macro'),
            'micro_f1_train': f1_score(y_train, y_train_pred, average='micro'),
            'micro_f1_valid': f1_score(y_valid, y_valid_pred, average='micro'),

            'macro_precision_train': precision_score(y_train, y_train_pred, average='macro'),
            'macro_precision_valid': precision_score(y_valid, y_valid_pred, average='macro'),
            'micro_precision_train': precision_score(y_train, y_train_pred, average='micro'),
            'micro_precision_valid': precision_score(y_valid, y_valid_pred, average='micro'),

            'macro_recall_train': recall_score(y_train, y_train_pred, average='macro'),
            'macro_recall_valid': recall_score(y_valid, y_valid_pred, average='macro'),
            'micro_recall_train': recall_score(y_train, y_train_pred, average='micro'),
            'micro_recall_valid': recall_score(y_valid, y_valid_pred, average='micro'),

            'macro_rocauc_train': roc_auc_score(y_train, y_train_pred_proba, average='macro', multi_class='ovr'),
            'macro_rocauc_valid': roc_auc_score(y_valid, y_valid_pred_proba, average='macro', multi_class='ovr'),
            'weighted_rocauc_train': roc_auc_score(y_train, y_train_pred_proba, average='weighted', multi_class='ovr'),
            'weighted_rocauc_valid': roc_auc_score(y_valid, y_valid_pred_proba, average='weighted', multi_class='ovr'),

            'confusion_matrix_train': confusion_matrix(y_train, y_train_pred),
            'confusion_matrix_valid': confusion_matrix(y_valid, y_valid_pred),
        }
    else:
        dict_metrics = {
            'vectorizer_time': end_time_vectorizer - start_time_vectorizer,
            'model_time': end_time_model - start_time_model,
            
            'vocab_size': len(vectorizer.vocabulary_.keys()),
            
            'accuracy_train': accuracy_score(y_train, y_train_pred),
            'accuracy_valid': accuracy_score(y_valid, y_valid_pred),
            
            'log_loss_train': log_loss(y_train, y_train_pred),
            'log_loss_valid': log_loss(y_valid, y_valid_pred),

            'f1_train': f1_score(y_train, y_train_pred),
            'f1_valid': f1_score(y_valid, y_valid_pred),

            'precision_train': precision_score(y_train, y_train_pred),
            'precision_valid': precision_score(y_valid, y_valid_pred),

            'recall_train': recall_score(y_train, y_train_pred),
            'recall_valid': recall_score(y_valid, y_valid_pred),

            'rocauc_train': roc_auc_score(y_train, y_train_pred),
            'rocauc_valid': roc_auc_score(y_valid, y_valid_pred),
        }

    # Predictions dict
    #dict_predictions = {'y_train_pred':y_train_pred, 'y_valid_pred':y_valid_pred}
    dict_predictions = {}

    return dict_metrics, dict_predictions


def train_all_folds(OUTPUT_FILENAME, CORPUS_FILE, CLASSIFIER, NCLASSES=2, VECTORIZER_TYPE='count', NGRAM_RANGE_=(1,1), VOCAB_SIZE=None):
    
    # Import dataset
    DATASET = pd.read_csv(os.path.join('0_data','datasets','b2w_preprocessed.csv'),low_memory=False)[['overall_rating','recommend_to_a_friend','kfold_2','kfold_5']]
    
    # Transform 2-class target feature to 0 and 1
    if NCLASSES==2:
        target_feature = {2:'recommend_to_a_friend', 5:'overall_rating'}[NCLASSES]
        DATASET[target_feature] = DATASET[target_feature].replace({'Yes':1,'No':0})

    # Import tokenized_file
    with open(os.path.join('0_data','datasets',CORPUS_FILE), 'r') as f:
        CORPUS = []
        for l in f:
            line = l.replace('\n','').split(',')
            CORPUS.append(line)
    
    # Create empty dict results
    overall_results = {}
    
    # Fit each fold
    #print(f'Start training')
    for fold_i in range(5):
        dict_metrics_, dict_predictions_ = train_fold(dataset=DATASET, corpus=CORPUS, classifier=CLASSIFIER, fold=fold_i, nclasses=NCLASSES, vectorizer_type=VECTORIZER_TYPE, ngram_range_=NGRAM_RANGE_, vocab_size=VOCAB_SIZE)
        overall_results[fold_i] = {'metrics':dict_metrics_, 'predictions':dict_predictions_}
    #print(f'Finish training\n')
        
    # Fill dict results with some metadata
    overall_results['nclasses'] = NCLASSES
    overall_results['vectorizer_type'] = VECTORIZER_TYPE
    overall_results['ngram_range_'] = NGRAM_RANGE_
    overall_results['classifier_params'] = CLASSIFIER.get_params()
    
    # Save results
    joblib.dump(overall_results, os.path.join('0_data','models','feature_based',OUTPUT_FILENAME + '.joblib'))