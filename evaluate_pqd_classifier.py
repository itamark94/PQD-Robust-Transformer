import os
import torch
from torch.utils.data import DataLoader, TensorDataset

from models import Transformer1D, CNN, DeepCNN
from evaluator import evaluate_classification_performance
from utils import numpy_to_torch
from utils_plot import plot_confusion_matrix


def evaluate_pqd_classifier(name, x_test, y_test, gt_test, n_classes, le, device, args):
    """
    Evaluates classification performance of a PQD classifier in terms of accuracy, f1-score and confusion matrix (plots
    and saves the figure of the confusion matrix).

    Args:
        name: type of the PQD classifier (Transformer or DeepCNN).
        x_test: inputs of test-set.
        y_test: labels of test-set.
        gt_test: ground truth explanation vectors of test-set.
        n_classes: number of classes (types of PQDs including normal).
        le: label encoding transformation function.
        device: cuda or cpu.
        args: arguments (architecture properties and testing parameters).
    """
    x_test, y_test, gt_test = numpy_to_torch(x_test, y_test, gt_test)
    test_loader = DataLoader(TensorDataset(x_test, y_test), args.batch_size_test, shuffle=False)
    if name == 'CNN':
        model = CNN(n_classes)
    elif name == 'DeepCNN':
        model = DeepCNN(n_classes)
    else:
        model = Transformer1D(d_model=args.d_model,
                              nhead=args.nhead,
                              dim_feedforward=args.dim_feedforward,
                              dropout=args.dropout,
                              dropout_attn=args.dropout_attn,
                              n_blocks=args.n_blocks,
                              n_classes=n_classes,
                              return_attn_maps=args.exp)
    model.to(device)
    path_model = 'models\\' + name + '_model.pth'
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), path_model)))
    print("Testing " + name + " with no adversarial attack")
    acc, f1, cm = evaluate_classification_performance(model, test_loader, device)
    print(f'accuracy = {acc * 100:.2f}%, f1 = {f1 * 100:.2f}%')
    plot_confusion_matrix(cm, name, le, n_classes)
    if name == 'Transformer':
        for lambda_class in args.lambda_class:
            name = 'Transformer' + '_lambda_' + str(int(lambda_class*10))
            path_model = 'models\\' + name + '_model.pth'
            model.load_state_dict(torch.load(os.path.join(os.getcwd(), path_model)))
            acc, f1, cm = evaluate_classification_performance(model, test_loader, device)
            print(f'accuracy = {acc * 100:.2f}%, f1 = {f1 * 100:.2f}%')
            plot_confusion_matrix(cm, name, le, n_classes)
