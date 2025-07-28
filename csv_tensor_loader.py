# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: /home/muyi/shell_competition/csv_tensor_loader.py
# Bytecode version: 3.12.0rc2 (3531)
# Source timestamp: 2025-07-18 00:56:55 UTC (1752800215)

import pandas as pd
import tensorflow as tf
import numpy
import csv
import os

def reshape_features(features):
    features = tf.convert_to_tensor(features)
    return tf.reshape(features, (-1, 5, 11, 1))

def csv_to_tf_dataset(csv_filepath, batch_size=32, shuffle=True, test_size=0.2, random_state=30):
    """
    Reads a CSV file and converts it into TensorFlow tf.data.Datasets for training and testing.
    Automatically determines feature and label columns:
    - The last 10 columns are treated as label columns.
    - All other columns are treated as feature columns.

    Args:
        csv_filepath (str): The path to the CSV file.
        batch_size (int): The desired batch size for the dataset. Defaults to 32.
        shuffle (bool): Whether to shuffle the dataset. Defaults to True.
        test_size (float): Fraction of data to use as test set. Defaults to 0.2.
        random_state (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        (tf.data.Dataset, tf.data.Dataset): Tuple of (train_dataset, test_dataset)
    """
    try:
        df = pd.read_csv(csv_filepath)
        all_columns = df.columns.tolist()
        if len(all_columns) < 10:
            print('Error: CSV file has fewer than 10 columns, cannot determine last 10 as labels.')
            return (None, None)
        label_columns = all_columns[-10:]
        feature_columns = all_columns[:-10]
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        test_count = int(len(df) * test_size)
        test_df = df.iloc[:test_count]
        train_df = df.iloc[test_count:]
        train_features = train_df[feature_columns]
        train_labels = train_df[label_columns]
        test_features = test_df[feature_columns]
        test_labels = test_df[label_columns]

        train_feature_tensors = tf.constant(train_features.values, dtype=tf.float32)
        train_label_tensors = tf.constant(train_labels.values, dtype=tf.float32)
        test_feature_tensors = tf.constant(test_features.values, dtype=tf.float32)
        test_label_tensors = tf.constant(test_labels.values, dtype=tf.float32)
        train_dataset = tf.data.Dataset.from_tensor_slices((train_feature_tensors, train_label_tensors))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_feature_tensors, test_label_tensors))

        def map_fn(features, labels):
            features = tf.reshape(features, (5, 11, 1))
            return (features, labels)
        
        train_dataset = train_dataset.map(map_fn)
        test_dataset = test_dataset.map(map_fn)
        
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=len(train_df))
        train_dataset = train_dataset.batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)
        return (train_dataset, test_dataset)
    except FileNotFoundError:
        print(f"Error: The file '{csv_filepath}' was not found.")
        return (None, None)
    
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
        return (None, None) + (None, e) * e
    
if __name__ == '__main__':
    csv_path = 'dataset/train.csv'
    try:
        df = pd.read_csv(csv_path)
        print('Loaded DataFrame:')
        print(df)
    except Exception as e:
        pass
    else:
        print('\nTrain/Test TensorFlow Datasets:')
        train_ds, test_ds = csv_to_tf_dataset(csv_path)
        print('Train Dataset:', train_ds)
        print('Test Dataset:', test_ds)
    print(f'Failed to load DataFrame: {e}')