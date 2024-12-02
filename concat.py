from pandas import DataFrame
import numpy as np
import pandas as pd
import csv
import os

# -----------------------------
# Function Definitions
# -----------------------------

def chang_to_array(s):
    """
    Processes the input data by cleaning strings, converting them to floats,
    and concatenating the first two integer values with the feature vectors.

    Parameters:
    - s (numpy.ndarray): Input data array.

    Returns:
    - numpy.ndarray: Processed data array with shape (num_samples, 2 + feature_length).
    """
    final_model = []
    for idx, i in enumerate(s):
        try:
            ev = int(i[0])
            dr = int(i[1])
            h = i[2].replace('[', "").replace('...', "") \
                   .replace('\n', "").replace(']', "") \
                   .replace('  ', " ").strip().split(" ")
            # Convert to floats, removing any empty strings
            h = [float(d) for d in h if d]
            h = np.array(h, dtype=np.float64)
            # Concatenate ev, dr with h
            con = np.concatenate(([ev, dr], h))
            final_model.append(con)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue  # Skip problematic rows
    return np.array(final_model)

def reduc_shape(m):
    """
    Aggregates feature vectors for each unique index by computing their mean.

    Parameters:
    - m (numpy.ndarray): Input data array with at least three columns.

    Returns:
    - list of lists: Each sublist contains an index and its aggregated feature vector.
    """
    r = []
    for i in range(572):
        s2 = np.where(m[:, 1] == i)[0]
        if len(s2) == 0:
            # No entries found for this index; optionally, handle as needed
            continue
        # Aggregate using mean to maintain consistent feature size
        dd = np.mean(m[s2, 2:], axis=0)
        r.append([i, dd])
    return r

def make_dic(x):
    """
    Converts a list of [index, feature_vector] into a dictionary.

    Parameters:
    - x (list of lists): Each sublist contains an index and its feature vector.

    Returns:
    - dict: Mapping from index to feature vector.
    """
    s_dic = {}
    for item in x:
        index = int(item[0])
        feature_vector = item[1]
        s_dic[index] = feature_vector
    return s_dic

def multiply_features(f_i, DDI):
    """
    Multiplies feature vectors for pairs of indices specified in DDI.

    Parameters:
    - f_i (dict): Dictionary mapping indices to their feature vectors.
    - DDI (numpy.ndarray): Array of index pairs.

    Returns:
    - numpy.ndarray: Array of multiplied feature vectors.
    """
    features = []
    for idx, d in enumerate(DDI):
        d0, d1 = int(d[0]), int(d[1])
        feat0 = f_i.get(d0, np.zeros_like(next(iter(f_i.values()), [])))
        feat1 = f_i.get(d1, np.zeros_like(next(iter(f_i.values()), [])))
        feature = np.multiply(feat0, feat1)
        features.append(feature)
        if idx < 5:  # Debug: Print first few feature multiplications
            print(f"Multiplying indices {d0} and {d1}: {feature}")
    return np.array(features)

# -----------------------------
# Configuration: Define Paths
# -----------------------------

# Base directory
base_dir = "/kaggle/working/GNN_DDI/DDI/data5/"

# Input CSV file paths
file_paths = [
    os.path.join(base_dir, "final_modelss1.csv"),
    os.path.join(base_dir, "final_modelss2.csv"),
    os.path.join(base_dir, "final_modelss3.csv"),
    os.path.join(base_dir, "final_modelss4.csv")
]

# Output file paths
output_paths = [
    os.path.join(base_dir, "t_c_m_1_32.txt"),
    os.path.join(base_dir, "t_c_m_2_32.txt"),
    os.path.join(base_dir, "t_c_m_3_32.txt"),
    os.path.join(base_dir, "t_c_m_4_32.txt")
]

# Path to full_pos.txt
full_pos_path = os.path.join(base_dir, "full_pos.txt")

# -----------------------------
# Data Loading and Processing
# -----------------------------

# Process each input file using chang_to_array
processed_data = []
for path in file_paths:
    try:
        data = pd.read_csv(path, header=None, sep=',').values
        array = chang_to_array(data)
        print(f"{path} shape after processing: {array.shape}")
        processed_data.append(array)
    except Exception as e:
        print(f"Error loading {path}: {e}")

# Unpack processed data
if len(processed_data) != 4:
    print("Error: Not all input files were processed successfully.")
    # Handle the error as needed
else:
    x1, x2, x3, x4 = processed_data

    # Reduce shape with aggregation
    reduced_data = []
    for idx, x in enumerate([x1, x2, x3, x4], start=1):
        r = reduc_shape(x)
        reduced_data.append(r)
        print(f"Reduced data {idx} length: {len(r)}")

    # Convert reduced data to dictionaries
    xs1, xs2, xs3, xs4 = [make_dic(r) for r in reduced_data]
    print(f"Length of xs1: {len(xs1)}")

    # Combine all feature dictionaries into a list (not converting to np.array)
    all_features = [xs1, xs2, xs3, xs4]
    print(f"Number of feature dictionaries: {len(all_features)}")
    print(f"First 5 keys in xs1: {list(xs1.keys())[:5]}")

    # -----------------------------
    # DDI Data Processing
    # -----------------------------

    try:
        full_pos = pd.read_csv(full_pos_path, header=None, sep=' ').values
        print(f"full_pos shape: {full_pos.shape}")
    except Exception as e:
        print(f"Error loading full_pos.txt: {e}")
        full_pos = np.array([])

    if full_pos.size > 0:
        DDI = full_pos[:, 1:3]
        print(f"DDI shape: {DDI.shape}")
    else:
        DDI = np.array([])
        print("DDI data is empty.")

    # -----------------------------
    # Feature Multiplication and Saving
    # -----------------------------

    if DDI.size > 0:
        # Multiply features for each feature dictionary
        new_feature1 = multiply_features(xs1, DDI)
        new_feature2 = multiply_features(xs2, DDI)
        new_feature3 = multiply_features(xs3, DDI)
        new_feature4 = multiply_features(xs4, DDI)

        multiplied_features = [new_feature1, new_feature2, new_feature3, new_feature4]

        # Save multiplied features to CSV
        for feature, path in zip(multiplied_features, output_paths):
            try:
                pd.DataFrame(feature).to_csv(path, header=None, index=None, sep=' ')
                print(f"Saved multiplied features to {path} with shape {feature.shape}")
            except Exception as e:
                print(f"Error saving to {path}: {e}")
    else:
        print("No DDI data to process.")

    # -----------------------------
    # Cleanup (Optional)
    # -----------------------------

    # Free up memory by deleting large variables
    del x1, x2, x3, x4, reduced_data, xs1, xs2, xs3, xs4, all_features, full_pos, DDI
    print("Cleanup completed. Large variables have been deleted.")
