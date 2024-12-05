import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils import numpy_to_torch
from models import Transformer1D, CNN, DeepCNN
from trainer import train


def train_pqd_classifier_transformer(x_train, y_train, gt_train, x_val, y_val, gt_val, n_classes, args, device):
    """
    Trains a Transformer PQD classifier. In case you wish to train a new Transformer PQD classifier, first train with
    args.exp=False. Then, if you wish to fine-tune the Transformer using the proposed procedure in the
    paper, call this function while args.exp=True.

    Args:
        x_train: inputs of train-set.
        y_train: labels of train-set.
        gt_train: ground truth explanation vectors of train-set.
        x_val: inputs of validation-set.
        y_val: labels of validation-set.
        gt_val: ground truth explanation vectors of validation-set.
        n_classes: number of classes (types of PQDs including normal).
        args: arguments (architecture properties and training parameters).
        device: cuda of cpu.
    """
    x_train, y_train, gt_train = numpy_to_torch(x_train, y_train, gt_train)
    x_val, y_val, gt_val = numpy_to_torch(x_val, y_val, gt_val)
    gt_train = torch.softmax(gt_train, dim=-1)
    gt_val = torch.softmax(gt_val, dim=-1)

    name = "Transformer"
    model = Transformer1D(d_model=args.d_model,
                          nhead=args.nhead,
                          dim_feedforward=args.dim_feedforward,
                          dropout=args.dropout,
                          dropout_attn=args.dropout_attn,
                          n_blocks=args.n_blocks,
                          n_classes=n_classes,
                          return_attn_maps=args.exp)
    model.to(device)
    train_loader = DataLoader(TensorDataset(x_train, y_train, gt_train), args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val, gt_val), args.batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr if args.exp else args.lr_exp)
    criteria = [nn.CrossEntropyLoss(), nn.MSELoss()]
    if args.exp:
        for lambda_class in args.lambda_class:
            train(model, name, train_loader, val_loader, optimizer, criteria, args, lambda_class)
    else:
        train(model, name, train_loader, val_loader, optimizer, criteria, args, None)


def train_pqd_classifier_cnn(x_train, y_train, gt_train, x_val, y_val, gt_val, n_classes, args, device, surrogate=False):
    """
    Trains a DeepCNN or a CNN PQD classifier. Make sure args.exp=False when calling this function.

    Args:
        x_train: inputs of train-set.
        y_train: labels of train-set.
        gt_train: ground truth explanation vectors of train-set.
        x_val: inputs of validation-set.
        y_val: labels of validation-set.
        gt_val: ground truth explanation vectors of validation-set.
        n_classes: number of classes (types of PQDs including normal).
        args: arguments (architecture properties and training parameters).
        device: cuda of cpu.
        surrogate: determines whether to train CNN as a surrogate model for black-box attacks.
    """
    x_train, y_train, gt_train = numpy_to_torch(x_train, y_train, gt_train)
    x_val, y_val, gt_val = numpy_to_torch(x_val, y_val, gt_val)
    gt_train = torch.softmax(gt_train, dim=-1)
    gt_val = torch.softmax(gt_val, dim=-1)
    if surrogate:
        name = 'CNN'
        model = CNN(n_classes)
    else:
        name = 'DeepCNN'
        model = DeepCNN(n_classes)
    model.to(device)
    train_loader = DataLoader(TensorDataset(x_train, y_train, gt_train), args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val, gt_val), args.batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr if args.exp else args.lr_exp)
    criteria = [nn.CrossEntropyLoss(), nn.MSELoss()]
    if args.exp:
        raise 'Training CNN/DeepCNN PQD classifier is possible only when args.exp=False'
    else:
        train(model, name, train_loader, val_loader, optimizer, criteria, args, None)
