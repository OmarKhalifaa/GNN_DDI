# -----------------------------
# Import Statements
# -----------------------------
from numpy.random import seed as np_seed
np_seed(1)

# Uncomment and set TensorFlow seed if needed
# from tensorflow import set_random_seed
# set_random_seed(2)

import os  # Ensure os is imported
import csv
import sqlite3
import time
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import (
    auc,
    roc_auc_score,
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    precision_recall_curve,
    roc_curve
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Model
from keras.layers import (
    Dense, Dropout, Input, Activation, BatchNormalization, LSTM, 
    MaxPooling1D, Conv1D
)
from keras.callbacks import EarlyStopping
import tensorflow as tf

# -----------------------------
# Configuration: Define Paths and Parameters
# -----------------------------

# Base directory (parent of data5 and full_pos2.txt)
base_dir = "/kaggle/working/GNN_DDI/DDI/"

# Data5 subdirectory
data5_dir = os.path.join(base_dir, "data5/")

# Input CSV file paths within data5
file_paths = [
    os.path.join(data5_dir, "final_modelssd1_d_32.csv"),
    os.path.join(data5_dir, "final_modelssd2_d_32.csv"),
    os.path.join(data5_dir, "final_modelssd3_d_32.csv"),
    os.path.join(data5_dir, "final_modelssd4_d_32.csv")
]

# Output file paths within data5
output_paths = [
    os.path.join(data5_dir, "t_c_m_1_32.txt"),
    os.path.join(data5_dir, "t_c_m_2_32.txt"),
    os.path.join(data5_dir, "t_c_m_3_32.txt"),
    os.path.join(data5_dir, "t_c_m_4_32.txt")
]

# Path to full_pos2.txt in base_dir
full_pos_path = os.path.join(base_dir, "full_pos2.txt")

# Feature-related parameters
event_num = 65
droprate = 0.3
vector_size = 32  # Updated to match feature dimension
clf = "DDIMDL"
CV = 5
seed_value = 0  # Renamed to avoid conflict with function 'seed'
f_matrix = [1, 2, 3, 4]
featureName = os.path.join(data5_dir, "G_allf_32_cm")

# -----------------------------
# Function Definitions
# -----------------------------

def DNN():
    """
    Defines and compiles a Deep Neural Network model.
    
    Returns:
        keras.Model: Compiled DNN model.
    """
    train_input = Input(shape=(vector_size,), name='Inputlayer')  # Now (32,)
    train_in = Dense(512, activation='relu')(train_input)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)
    train_in = Dense(256, activation='relu')(train_in)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)
    train_in = Dense(event_num)(train_in)
    out = Activation('softmax')(train_in)
    model = Model(inputs=train_input, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_index(label_matrix, event_num, seed, CV):
    """
    Generates cross-validation indices for each class.
    
    Parameters:
        label_matrix (numpy.ndarray): Array of labels.
        event_num (int): Number of unique events/classes.
        seed (int): Random seed for reproducibility.
        CV (int): Number of cross-validation splits.
    
    Returns:
        numpy.ndarray: Array indicating the fold index for each sample.
    """
    index_all_class = np.zeros(len(label_matrix), dtype=int)
    for j in range(event_num):
        indices = np.where(label_matrix == j)[0]
        if len(indices) == 0:
            continue  # Skip if no samples for this class
        kf = KFold(n_splits=CV, shuffle=True, random_state=seed)
        for k_num, (train_index, test_index) in enumerate(kf.split(indices)):
            index_all_class[indices[test_index]] = k_num
    return index_all_class

def bring_f(f_item):
    """
    Loads and processes feature data from a specific file.
    
    Parameters:
        f_item (str): Feature identifier (e.g., '1', '2', '3', '4').
    
    Returns:
        list: List of feature vectors.
    """
    file_path = os.path.join(data5_dir, f"t_c_m_{f_item}_32.txt")
    try:
        full_dataframe = pd.read_csv(file_path, header=None, sep=' ')
        x1 = full_dataframe.values  # More efficient than np.array(full_dataframe).tolist()
        print(f"Loaded features from {file_path}: {x1.shape}")
        return x1.tolist()
    except Exception as e:
        print(f"Error loading features from {file_path}: {e}")
        return []

def initialize_classifier(clf_type):
    """
    Initializes the classifier based on the specified type.
    
    Parameters:
        clf_type (str): Type of classifier to initialize.
    
    Returns:
        sklearn classifier: Initialized classifier.
    """
    if clf_type == 'RF':
        return RandomForestClassifier(n_estimators=100, random_state=seed_value)
    elif clf_type == 'GBDT':
        return GradientBoostingClassifier(random_state=seed_value)
    elif clf_type == 'SVM':
        return SVC(probability=True, random_state=seed_value)
    elif clf_type == 'FM':
        return GradientBoostingClassifier(random_state=seed_value)  # Replace with actual FM if different
    elif clf_type == 'KNN':
        return KNeighborsClassifier(n_neighbors=4)
    else:
        return LogisticRegression(max_iter=1000, random_state=seed_value)

def cross_validation(feature_matrix, label_matrix, clf_type, event_num, seed, CV):
    """
    Performs cross-validation using the specified classifier.
    
    Parameters:
        feature_matrix (list or list of lists): List of feature identifiers.
        label_matrix (numpy.ndarray): Array of labels.
        clf_type (str): Type of classifier to use.
        event_num (int): Number of unique events/classes.
        seed (int): Random seed for reproducibility.
        CV (int): Number of cross-validation splits.
    
    Returns:
        tuple: Predicted labels, prediction scores, and true labels.
    """
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    y_true = np.array([])
    y_pred = np.array([])
    y_score = np.zeros((0, event_num), dtype=float)
    index_all_class = get_index(label_matrix, event_num, seed, CV)
    
    for k in range(CV):
        print(f"Cross-Validation Fold {k+1}/{CV}")
        train_index = np.where(index_all_class != k)[0]
        test_index = np.where(index_all_class == k)[0]
        pred = np.zeros((len(test_index), event_num), dtype=float)
        
        for i, f_item in enumerate(feature_matrix):
            print(f"Processing Feature {i+1}/{len(feature_matrix)}: {f_item}")
            features = bring_f(str(f_item))
            if not features:
                continue  # Skip if no features loaded
            xx = np.array(features)
            x_train = xx[train_index]
            x_test = xx[test_index]
            y_train = label_matrix[train_index]
            y_test = label_matrix[test_index]
            
            # One-hot encoding
            y_train_one_hot = label_binarize(y_train, classes=np.arange(event_num))
            y_test_one_hot = label_binarize(y_test, classes=np.arange(event_num))
            
            if clf_type == 'DDIMDL':
                dnn = DNN()
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
                dnn.fit(
                    x_train, y_train_one_hot, 
                    batch_size=128, epochs=100,
                    validation_data=(x_test, y_test_one_hot),
                    callbacks=[early_stopping],
                    verbose=0  # Suppress training output
                )
                pred += dnn.predict(x_test, verbose=0)
                del dnn  # Free memory
            else:
                # Initialize the specified classifier
                clf = initialize_classifier(clf_type)
                clf.fit(x_train, y_train)
                pred += clf.predict_proba(x_test)
        
        pred_score = pred / len(feature_matrix)  # Average predictions
        pred_type = np.argmax(pred_score, axis=1)
        y_true = np.hstack((y_true, label_matrix[test_index]))
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.vstack((y_score, pred_score))
    
    return y_pred, y_score, y_true

def evaluate(pred_type, pred_score, y_test, event_num):
    """
    Evaluates the performance of the predictions.
    
    Parameters:
        pred_type (numpy.ndarray): Predicted class labels.
        pred_score (numpy.ndarray): Prediction scores/probabilities.
        y_test (numpy.ndarray): True class labels.
        event_num (int): Number of unique events/classes.
    
    Returns:
        tuple: Overall results and per-event results.
    """
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    
    y_one_hot = label_binarize(y_test, classes=np.arange(event_num))
    pred_one_hot = label_binarize(pred_type, classes=np.arange(event_num))
    
    # Overall Metrics
    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all[2] = roc_aupr_score(y_one_hot, pred_score, average='macro')
    result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all[4] = roc_auc_score(y_one_hot, pred_score, average='macro')
    result_all[5] = f1_score(y_test, pred_type, average='micro')
    result_all[6] = f1_score(y_test, pred_type, average='macro')
    result_all[7] = precision_score(y_test, pred_type, average='micro')
    result_all[8] = precision_score(y_test, pred_type, average='macro')
    result_all[9] = recall_score(y_test, pred_type, average='micro')
    result_all[10] = recall_score(y_test, pred_type, average='macro')
    
    # Per-Event Metrics
    for i in range(event_num):
        y_true_event = y_one_hot[:, i]
        y_pred_event = pred_one_hot[:, i]
        if len(np.unique(y_true_event)) > 1:
            result_eve[i, 0] = accuracy_score(y_true_event, y_pred_event)
            result_eve[i, 1] = roc_aupr_score(y_true_event, pred_score[:, i], average=None)
            result_eve[i, 2] = roc_auc_score(y_true_event, pred_score[:, i], average=None)
            result_eve[i, 3] = f1_score(y_true_event, y_pred_event, average='binary')
            result_eve[i, 4] = precision_score(y_true_event, y_pred_event, average='binary')
            result_eve[i, 5] = recall_score(y_true_event, y_pred_event, average='binary')
        else:
            # Handle cases with only one class present
            result_eve[i, :] = np.nan  # Or any other placeholder
    
    return [result_all, result_eve]

def roc_aupr_score(y_true, y_score, average="macro"):
    """
    Calculates the AUPR score for binary or multiclass classification.
    
    Parameters:
        y_true (numpy.ndarray): True binary labels in one-hot format.
        y_score (numpy.ndarray): Predicted scores/probabilities.
        average (str): Averaging method ('binary', 'micro', 'macro').
    
    Returns:
        float: AUPR score.
    """
    def _binary_roc_aupr_score(y_true_bin, y_score_bin):
        precision, recall, _ = precision_recall_curve(y_true_bin, y_score_bin)
        return auc(recall, precision)

    def _average_binary_score(binary_metric, y_true, y_score, average):
        if average == "binary":
            return binary_metric(y_true[:, 0], y_score[:, 0])
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true[:, c]
            y_score_c = y_score[:, c]
            if len(np.unique(y_true_c)) > 1:
                score[c] = binary_metric(y_true_c, y_score_c)
            else:
                score[c] = np.nan  # Undefined AUPR
        if average == "macro":
            return np.nanmean(score)
        elif average == "micro":
            return np.nanmean(score)
        else:
            return score

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)

def save_result(feature_name, result_type, clf_type, result):
    """
    Saves the evaluation results to a CSV file.
    
    Parameters:
        feature_name (str): Base name for the feature.
        result_type (str): Type of result ('all' or 'each').
        clf_type (str): Classifier type.
        result (numpy.ndarray): Evaluation results.
    
    Returns:
        None
    """
    filename = f"{feature_name}_{result_type}_{clf_type}.csv"
    try:
        np.savetxt(filename, result, delimiter=",", fmt='%.6f')
        print(f"Saved {result_type} results to {filename}")
    except Exception as e:
        print(f"Error saving results to {filename}: {e}")

# -----------------------------
# Main Execution
# -----------------------------

def main():
    # Start timing using time.process_time()
    start = time.process_time()
    
    # Load full_pos2.txt
    try:
        full_pos = pd.read_csv(full_pos_path, header=None, sep=' ').values
        print(f"Loaded full_pos2.txt with shape: {full_pos.shape}")
    except Exception as e:
        print(f"Error loading {full_pos_path}: {e}")
        return  # Exit if loading fails
    
    # Extract labels
    new_label = full_pos[:, 0].astype(int)
    print(f'new_label: {len(new_label)}, first label: {new_label[0]}')
    
    # Extract DDI pairs (not used in main, but kept for consistency)
    DDI = full_pos[:, 1:3].astype(int)
    new_label = np.array(new_label)
    
    # Perform cross-validation
    y_pred, y_score, y_true = cross_validation(
        feature_matrix=f_matrix, 
        label_matrix=new_label, 
        clf_type=clf, 
        event_num=event_num, 
        seed=seed_value, 
        CV=CV
    )
    
    # Evaluate predictions
    all_result, each_result = evaluate(y_pred, y_score, y_true, event_num)
    print("Overall Evaluation Results:\n", all_result)
    print("Per-Event Evaluation Results:\n", each_result)
    
    # Save results
    save_result(featureName, 'all', clf, all_result)
    save_result(featureName, 'each', clf, each_result)
    
    # End timing and print elapsed time
    elapsed_time = time.process_time() - start
    print(f"Time used: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
