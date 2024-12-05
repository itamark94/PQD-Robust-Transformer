import os
from time import time
import torch
from torch.utils.data import DataLoader, TensorDataset

import myattacks
from evaluator import evaluate_classification_performance, average_robustness, fooling_rate
from attacker import attack
from models import Transformer1D, CNN, DeepCNN
from utils import numpy_to_torch
from utils_plot import plot_confusion_matrix


def evaluate_black_box_attack(name, atk_name, x_test, y_test, n_classes, le, device, args):
    """
    Performs black-box adversarial attack and evaluates PQD classification performance in terms of accuracy, f1-score,
    average robustness, fooling rate and confusion matrix.

    Args:
        name: architecture of the PQD classifier: Transformer or DeepCNN.
        atk_name: attack method.
        x_test: inputs of test-set.
        y_test: labels of test-set.
        n_classes: number of classes.
        le: label encoding transformation function.
        device: cuda or cpu.
        args: arguments (architecture properties and training parameters).
    """
    x_test, y_test = numpy_to_torch(x_test, y_test, None)
    test_loader = DataLoader(TensorDataset(x_test, y_test), args.batch_size_test, shuffle=False)
    if name == 'DeepCNN':
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
    name_surrogate = 'CNN'
    model_surrogate = CNN(n_classes)
    model_surrogate.to(device)
    path_model_surrogate = 'models\\' + name_surrogate + '_model.pth'
    model_surrogate.load_state_dict(torch.load(os.path.join(os.getcwd(), path_model_surrogate)))
    robustness_list = []
    fooling_list = []
    acc_list = []
    f1_list = []
    atk_name_save = atk_name + '_black_box'

    for eps in args.eps_vec:
        atk_class = getattr(myattacks, atk_name)
        if atk_name == 'MIFGSM':
            atk = atk_class(model_surrogate, eps, alpha=eps / args.steps, steps=args.steps, decay=args.decay)
        elif atk_name == 'BIM' or atk_name == 'PGD':
            atk = atk_class(model_surrogate, eps, alpha=eps / args.steps, steps=args.steps)
        else:
            atk = atk_class(model_surrogate, eps)
        print('Attacking ' + name + ' using ' + atk_name_save + f' with epsilon = {eps}')
        start = time()
        x_adv, _ = attack(atk, test_loader, device)
        end = time()
        print(f"Attacking time = {end - start:.0f} seconds")
        robustness = average_robustness(x_test, x_adv)
        adv_test_loader = DataLoader(TensorDataset(x_adv, y_test), batch_size=args.batch_size_test, shuffle=False)
        fooling = fooling_rate(model, x_test, x_adv, args.batch_size_test, device)
        acc, f1, cm = evaluate_classification_performance(model, adv_test_loader, device)
        plot_confusion_matrix(cm, name + '_' + atk_name_save + '_eps_' + str(eps), le, n_classes)
        acc_list.append(acc)
        f1_list.append(f1)
        fooling_list.append(fooling)
        robustness_list.append(robustness)

    path_acc = 'scores\\' + atk_name_save + '_' + name + '_accuracy.pth'
    path_f1 = 'scores\\' + atk_name_save + '_' + name + '_f1.pth'
    path_robustness = 'scores\\' + atk_name_save + '_' + name + '_robustness.pth'
    path_fooling = 'scores\\' + atk_name_save + '_' + name + '_fooling.pth'
    torch.save(torch.FloatTensor(acc_list), os.path.join(os.getcwd(), path_acc))
    torch.save(torch.FloatTensor(f1_list), os.path.join(os.getcwd(), path_f1))
    torch.save(torch.FloatTensor(fooling_list), os.path.join(os.getcwd(), path_fooling))
    torch.save(torch.FloatTensor(robustness_list), os.path.join(os.getcwd(), path_robustness))

    if name == 'Transformer':
        for lambda_class in args.lambda_class:
            name = 'Transformer_lambda_' + str(int(lambda_class*10))
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
            robustness_list = []
            fooling_list = []
            acc_list = []
            f1_list = []
            atk_name_save = atk_name + '_black_box'
            for eps in args.eps_vec:
                atk_class = getattr(myattacks, atk_name)
                if atk_name == 'MIFGSM':
                    atk = atk_class(model_surrogate, eps, alpha=eps / args.steps, steps=args.steps, decay=args.decay)
                elif atk_name == 'BIM' or atk_name == 'PGD':
                    atk = atk_class(model_surrogate, eps, alpha=eps / args.steps, steps=args.steps)
                else:
                    atk = atk_class(model_surrogate, eps)
                print('Attacking ' + name + ' using ' + atk_name_save + f' with epsilon = {eps}')
                start = time()
                x_adv, _ = attack(atk, test_loader, device)
                end = time()
                print(f"Attacking time = {end - start:.0f} seconds")
                robustness = average_robustness(x_test, x_adv)
                adv_test_loader = DataLoader(TensorDataset(x_adv, y_test), batch_size=args.batch_size_test,
                                             shuffle=False)
                fooling = fooling_rate(model, x_test, x_adv, args.batch_size_test, device)
                acc, f1, cm = evaluate_classification_performance(model, adv_test_loader, device)
                plot_confusion_matrix(cm, name + '_' + atk_name_save + '_eps_' + str(eps), le, n_classes)
                acc_list.append(acc)
                f1_list.append(f1)
                fooling_list.append(fooling)
                robustness_list.append(robustness)

            path_acc = 'scores\\' + atk_name_save + '_' + name + '_accuracy.pth'
            path_f1 = 'scores\\' + atk_name_save + '_' + name + '_f1.pth'
            path_robustness = 'scores\\' + atk_name_save + '_' + name + '_robustness.pth'
            path_fooling = 'scores\\' + atk_name_save + '_' + name + '_fooling.pth'
            torch.save(torch.FloatTensor(acc_list), os.path.join(os.getcwd(), path_acc))
            torch.save(torch.FloatTensor(f1_list), os.path.join(os.getcwd(), path_f1))
            torch.save(torch.FloatTensor(fooling_list), os.path.join(os.getcwd(), path_fooling))
            torch.save(torch.FloatTensor(robustness_list), os.path.join(os.getcwd(), path_robustness))


def evaluate_white_box_attack(name, atk_name, x_test, y_test, n_classes, le, device, args):
    """
    Performs white-box adversarial attack and evaluates PQD classification performance in terms of accuracy, f1-score,
    average robustness, fooling rate and confusion matrix.

    Args:
        name: architecture of the PQD classifier: Transformer or DeepCNN.
        atk_name: attack method.
        x_test: inputs of test-set.
        y_test: labels of test-set.
        n_classes: number of classes.
        le: label encoding transformation function.
        device: cuda or cpu.
        args: arguments (architecture properties and training parameters).
    """
    x_test, y_test = numpy_to_torch(x_test, y_test)
    test_loader = DataLoader(TensorDataset(x_test, y_test), args.batch_size_test, shuffle=False)
    if name == 'DeepCNN':
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
    robustness_list = []
    fooling_list = []
    acc_list = []
    f1_list = []
    atk_name_save = atk_name + '_white_box'

    for eps in args.eps_vec:
        atk_class = getattr(myattacks, atk_name)
        if atk_name == 'MIFGSM':
            atk = atk_class(model, eps, alpha=eps / args.steps, steps=args.steps, decay=args.decay)
        elif atk_name == 'BIM' or atk_name == 'PGD':
            atk = atk_class(model, eps, alpha=eps / args.steps, steps=args.steps)
        else:
            atk = atk_class(model, eps)
        print('Attacking ' + name + ' using ' + atk_name_save + f' with epsilon = {eps}')
        start = time()
        x_adv, _ = attack(atk, test_loader, device)
        end = time()
        print(f"Attacking time = {end - start:.0f} seconds")
        robustness = average_robustness(x_test, x_adv)
        adv_test_loader = DataLoader(TensorDataset(x_adv, y_test),
                                     batch_size=args.batch_size_test,
                                     shuffle=False)
        fooling = fooling_rate(model, x_test, x_adv, args.batch_size_test, device)
        acc, f1, cm = evaluate_classification_performance(model, adv_test_loader, device)
        plot_confusion_matrix(cm, name + '_' + atk_name_save + '_eps_' + str(eps), le, n_classes)
        acc_list.append(acc)
        f1_list.append(f1)
        fooling_list.append(fooling)
        robustness_list.append(robustness)

    path_acc = 'scores\\' + atk_name_save + '_' + name + '_accuracy.pth'
    path_f1 = 'scores\\' + atk_name_save + '_' + name + '_f1.pth'
    path_robustness = 'scores\\' + atk_name_save + '_' + name + '_robustness.pth'
    path_fooling = 'scores\\' + atk_name_save + '_' + name + '_fooling.pth'
    torch.save(torch.FloatTensor(acc_list), os.path.join(os.getcwd(), path_acc))
    torch.save(torch.FloatTensor(f1_list), os.path.join(os.getcwd(), path_f1))
    torch.save(torch.FloatTensor(fooling_list), os.path.join(os.getcwd(), path_fooling))
    torch.save(torch.FloatTensor(robustness_list), os.path.join(os.getcwd(), path_robustness))

    if name == 'Transformer':
        for lambda_class in args.lambda_class:
            name = 'Transformer_lambda_' + str(int(lambda_class*10))
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
            robustness_list = []
            fooling_list = []
            acc_list = []
            f1_list = []
            atk_name_save = atk_name + '_white_box'
            for eps in args.eps_vec:
                atk_class = getattr(myattacks, atk_name)
                if atk_name == 'MIFGSM':
                    atk = atk_class(model, eps, alpha=eps / args.steps, steps=args.steps, decay=args.decay)
                elif atk_name == 'BIM' or atk_name == 'PGD':
                    atk = atk_class(model, eps, alpha=eps / args.steps, steps=args.steps)
                else:
                    atk = atk_class(model, eps)
                print('Attacking ' + name + ' using ' + atk_name_save + f' with epsilon = {eps}')
                start = time()
                x_adv, _ = attack(atk, test_loader, device)
                end = time()
                print(f"Attacking time = {end - start:.0f} seconds")
                robustness = average_robustness(x_test, x_adv)
                adv_test_loader = DataLoader(TensorDataset(x_adv, y_test),
                                             batch_size=args.batch_size_test,
                                             shuffle=False)
                fooling = fooling_rate(model, x_test, x_adv, args.batch_size_test, device)
                acc, f1, cm = evaluate_classification_performance(model, adv_test_loader, device)
                plot_confusion_matrix(cm, name + '_' + atk_name_save + '_eps_' + str(eps), le, n_classes)
                acc_list.append(acc)
                f1_list.append(f1)
                fooling_list.append(fooling)
                robustness_list.append(robustness)

            path_acc = 'scores\\' + atk_name_save + '_' + name + '_accuracy.pth'
            path_f1 = 'scores\\' + atk_name_save + '_' + name + '_f1.pth'
            path_robustness = 'scores\\' + atk_name_save + '_' + name + '_robustness.pth'
            path_fooling = 'scores\\' + atk_name_save + '_' + name + '_fooling.pth'
            torch.save(torch.FloatTensor(acc_list), os.path.join(os.getcwd(), path_acc))
            torch.save(torch.FloatTensor(f1_list), os.path.join(os.getcwd(), path_f1))
            torch.save(torch.FloatTensor(fooling_list), os.path.join(os.getcwd(), path_fooling))
            torch.save(torch.FloatTensor(robustness_list), os.path.join(os.getcwd(), path_robustness))
