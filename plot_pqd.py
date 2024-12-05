import os
import torch
from torch.utils.data import DataLoader, TensorDataset

from models import Transformer1D, DeepCNN
from utils import numpy_to_torch
from utils_plot import plot_pqd_natural_and_adversarial, plot_pqd_with_relevance_map, plot_pqd
import myattacks


def plot_pqd_with_relevance_map_main(label, x_test, y_test, n_classes, le, scaler, device, args):
    x_test, y_test = numpy_to_torch(x_test, y_test)
    indices = torch.where(y_test == le.transform([label])[0])
    x_test = x_test[indices]
    y_test = y_test[indices]
    test_loader = DataLoader(TensorDataset(x_test, y_test), args.batch_size_plot, shuffle=False)
    name = 'Transformer'
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

    name = 'Transformer_lambda_5'
    model_lambda = Transformer1D(d_model=args.d_model,
                                 nhead=args.nhead,
                                 dim_feedforward=args.dim_feedforward,
                                 dropout=args.dropout,
                                 dropout_attn=args.dropout_attn,
                                 n_blocks=args.n_blocks,
                                 n_classes=n_classes,
                                 return_attn_maps=args.exp)
    model_lambda.to(device)
    path_model = 'models\\' + name + '_model.pth'
    model_lambda.load_state_dict(torch.load(os.path.join(os.getcwd(), path_model)))

    model.eval()
    model_lambda.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            predictions, rollout = model.forward_and_attention_rollout(x)
            _, predicted = torch.max(predictions, dim=1)
            if(predicted == y):
                print('i = ', i)
                print('plot')
                vmax = plot_pqd_with_relevance_map(x[0], rollout[0], y[0], predicted[0], scaler, le)
                predictions, rollout = model_lambda.forward_and_attention_rollout(x)
                _, predicted = torch.max(predictions, dim=1)
                plot_pqd_with_relevance_map(x[0], rollout[0], y[0], predicted[0], scaler, le, vmax)
                print('-------------')


def plot_pqd_before_and_after_attack(label, x_test, y_test, n_classes, le, scaler, device, args):
    x_test, y_test = numpy_to_torch(x_test, y_test)
    indices = torch.where(y_test == le.transform([label])[0])
    x_test = x_test[indices]
    y_test = y_test[indices]
    test_loader = DataLoader(TensorDataset(x_test, y_test), args.batch_size_plot, shuffle=False)
    name = 'Transformer'
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
    atk_name = 'PGD'
    atk_name_save = atk_name + '_white_box'

    eps = 0.1
    atk_class = getattr(myattacks, atk_name)
    if atk_name == 'MIFGSM':
        atk = atk_class(model, eps, alpha=eps / args.steps, steps=args.steps, decay=args.decay)
    elif atk_name == 'BIM' or atk_name == 'PGD':
        atk = atk_class(model, eps, alpha=eps / args.steps, steps=args.steps)
    else:
        atk = atk_class(model, eps)
    print('Attacking ' + name + ' using ' + atk_name_save + f' with epsilon = {eps}')

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        x_adv = atk(x, y)
        _, yhat = model(x).max(1)
        _, yhat_adv = model(x_adv).max(1)



def plot_pqds_natural(x_test, y_test, le, scaler, labels_indices):
    x_test, y_test = numpy_to_torch(x_test, y_test)
    for i, index in enumerate(labels_indices):
        x = x_test[i * 500 + index]
        y = y_test[i * 500 + index]
        plot_pqd(x, y, scaler, le)


def plot_pqds_natural_adversarial(name, x_test, y_test, le, scaler, n_classes, labels_indices, device, args):
    x_test, y_test = numpy_to_torch(x_test, y_test)
    indices = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
    labels_indices_helper = [x + y for x, y in zip(labels_indices, indices)]
    labels_indices = torch.tensor(labels_indices_helper)
    x_test = torch.index_select(x_test, dim=0, index=labels_indices)
    y_test = torch.index_select(y_test, dim=0, index=labels_indices)
    batch_size = 10 if name == 'DeepCNN' else 1
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size, shuffle=False)

    if name == 'Transformer':
        name = 'Transformer_lambda_5'
        model = Transformer1D(d_model=args.d_model,
                              nhead=args.nhead,
                              dim_feedforward=args.dim_feedforward,
                              dropout=args.dropout,
                              dropout_attn=args.dropout_attn,
                              n_blocks=args.n_blocks,
                              n_classes=n_classes,
                              return_attn_maps=args.exp)
    elif name == 'DeepCNN':
        model = DeepCNN(n_classes)

    model.to(device)
    path_model = 'models\\' + name + '_model.pth'
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), path_model)))
    # name_surrogate = 'CNN'
    # model_surrogate = CNN(n_classes)
    # model_surrogate.to(device)
    # path_model_surrogate = 'models\\' + name_surrogate + '_model.pth'
    # model_surrogate.load_state_dict(torch.load(os.path.join(os.getcwd(), path_model_surrogate)))
    atk_name = 'PGD'
    atk_name_save = atk_name + '_white_box'
    eps = 0.15
    atk_class = getattr(myattacks, atk_name)
    atk = atk_class(model, eps, alpha=eps / args.steps, steps=args.steps)
    print('Attacking ' + name + ' using ' + atk_name_save + f' with epsilon = {eps}')

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        x_adv = atk(x, y)
        _, yhat = model(x).max(1)
        _, yhat_adv = model(x_adv).max(1)
        if name == 'DeepCNN':
            for i in range(10):
                plot_pqd_natural_and_adversarial(x[i], x_adv[i], y[i], yhat[i], yhat_adv[i], scaler, le, name)
        else:
            plot_pqd_natural_and_adversarial(x[0], x_adv[0], y[0], yhat[0], yhat_adv[0], scaler, le, name)
