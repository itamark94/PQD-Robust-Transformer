import os
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder


def make_folders():
    """
    Creates folders necessary for executing the scripts: saving models, results vectors for plots and figures.
    """
    if not os.path.exists('./accuracies'):
        os.mkdir('./accuracies')

    if not os.path.exists('./figures'):
        os.mkdir('./figures')

    if not os.path.exists('./losses'):
        os.mkdir('./losses')

    if not os.path.exists('./models'):
        os.mkdir('./models')

    if not os.path.exists('./scores'):
        os.mkdir('./scores')


def load_dataset_classes(path, labels_order):
    """
    Loads the dataset of power quality disturbances (PQDs).

    Args:
        path: dataset file location.
        labels_order: labels order used for encoding.

    Returns:
        x: power quality signals.
        y: power quality labels.
        gt: ground truth explanation vectors.
        le: label encoding transformation function.
    """
    labels = []
    for i, label in enumerate(labels_order):
        dataset_signals = loadmat(path + label + '_signals.mat')
        dataset_explains = loadmat(path + label + '_explains.mat')
        N = dataset_signals['signals'].shape[0]
        labels += [label] * N
        if i == 0:
            x = dataset_signals['signals']
            gt = dataset_explains['explains']
        else:
            x = np.concatenate((x, dataset_signals['signals']))
            gt = np.concatenate((gt, dataset_explains['explains']))

    le = CustomLabelEncoder(labels_order)
    y = np.array(le.fit_transform(labels), dtype='int64')

    return x, y, gt, le


def split_train_test(x, y, gt, train_split):
    """
    Splits the dataset to train-set and test-set.

    Args:
        x: power quality signals.
        y: power quality labels.
        gt: ground truth explanation vectors.
        train_split: ratio of the training-set to the whole dataset.
    """
    labels, counts = np.unique(y, return_counts=True)
    train_val_indices = []
    test_indices = []

    for i, label in enumerate(labels):
        train_val_indices_label = list(range(i * counts[label], i * counts[label] + int(train_split * counts[label])))
        train_val_indices += train_val_indices_label
        test_indices_label = list(range(i * counts[label] + int(train_split * counts[label]), (i + 1) * counts[label]))
        test_indices += test_indices_label

    x_train, y_train, gt_train = x[train_val_indices], y[train_val_indices], gt[train_val_indices]
    x_test, y_test, gt_test = x[test_indices], y[test_indices], gt[test_indices]

    return x_train, y_train, gt_train, x_test, y_test, gt_test


def split_train_val(x, y, gt, val_split, seed=None):
    """
    Splits the dataset to train-set and validation-set.

    Args:
        x: power quality signals.
        y: power quality labels.
        gt: ground truth explanation vectors.
        val_split: ratio of the validation-set to the training-set.
        seed: for reproducibility.
    """
    train_val_size = len(x)
    train_size = int((1 - val_split) * train_val_size)
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(train_val_size)
    x_train, y_train, gt_train = x[indices[:train_size]], y[indices[:train_size]], gt[indices[:train_size]]
    x_val, y_val, gt_val = x[indices[train_size:]], y[indices[train_size:]], gt[indices[train_size:]]
    np.random.seed()

    return x_train, y_train, gt_train, x_val, y_val, gt_val


def numpy_to_torch(x, y, gt=None):
    """
    Converts ndarrays (numpy) to tensors (torch).

    Args:
        x: power quality signals.
        y: power quality labels.
        gt: ground truth explanation vectors.
    """
    x = torch.from_numpy(np.expand_dims(x, axis=1)).float()
    y = torch.from_numpy(y)
    if gt is None:
        return x, y
    else:
        gt = torch.from_numpy(gt).float()
        return x, y, gt


class CustomLabelEncoder(LabelEncoder):
    def __init__(self, custom_order=None):
        self.custom_order = custom_order  # Our addition which allows encoding labels according to specific order
        super().__init__()

    def fit(self, y):
        if self.custom_order is None:
            super().fit(y)
        else:
            self.classes_ = self.custom_order

        return self

    def transform(self, y):
        if self.custom_order is None:
            return super().transform(y)
        else:
            return [self.classes_.index(label) for label in y]

    def inverse_transform(self, y):
        if self.custom_order is None:
            return super().inverse_transform(y)
        else:
            return [self.classes_[idx] for idx in y]

    def fit_transform(self, y):
        return self.fit(y).transform(y)
