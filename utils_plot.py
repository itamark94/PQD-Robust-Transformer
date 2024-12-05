import os
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, name, le, n_classes):
    """
    Plots confusion matrix of a certain PQD classifier and saves the figure.

    Args:
        cm: confusion matrix.
        name: type of the PQD classifier.
        le: encoder transform for converting labels from numbers to strings.
        n_classes: number of classes.
    """
    path = os.getcwd()+'\\figures\\' + 'confusion_matrix_' + name + '.pdf'
    cmp = ConfusionMatrixDisplay(cm, display_labels=le.inverse_transform(np.arange(n_classes)))
    fig, ax = plt.subplots(tight_layout=True)
    cmp.plot(include_values=True, cmap='Blues', ax=ax, xticks_rotation=40.0, colorbar=True)
    plt.savefig(path, format='pdf', bbox_inches='tight')
    plt.show()


def plot_metrics_vs_eps_diff_atks(atk_set, eps_vec):
    """
    Plots accuracy, F1 score, average robustness and fooling rate vs epsilon of different adversarial attacks on the
    Transformer for PQD classification.

    Args:
        atk_set: threat setting of the attack: 'black_box' or 'white_box'.
        eps_vec: vector of epsilon values.
    """
    path = os.getcwd() + '\\scores\\'

    name_ours = 'Transformer_lambda_5'  # with the proposed relevance fine-tuning procedure
    name_transformer = 'Transformer'

    fgsm_acc_ours = torch.load(path + 'FGSM_' + atk_set + '_' + name_ours + '_accuracy.pth')
    bim_acc_ours = torch.load(path + 'BIM_' + atk_set + '_' + name_ours + '_accuracy.pth')
    mifgsm_acc_ours = torch.load(path + 'MIFGSM_' + atk_set + '_' + name_ours + '_accuracy.pth')
    pgd_acc_ours = torch.load(path + 'PGD_' + atk_set + '_' + name_ours + '_accuracy.pth')
    fgsm_robustness_ours = torch.load(path + 'FGSM_' + atk_set + '_' + name_ours + '_robustness.pth')
    bim_robustness_ours = torch.load(path + 'BIM_' + atk_set + '_' + name_ours + '_robustness.pth')
    mifgsm_robustness_ours = torch.load(path + 'MIFGSM_' + atk_set + '_' + name_ours + '_robustness.pth')
    pgd_robustness_ours = torch.load(path + 'PGD_' + atk_set + '_' + name_ours + '_robustness.pth')

    fgsm_acc_transformer = torch.load(path + 'FGSM_' + atk_set + '_' + name_transformer + '_accuracy.pth')
    bim_acc_transformer = torch.load(path + 'BIM_' + atk_set + '_' + name_transformer + '_accuracy.pth')
    mifgsm_acc_transformer = torch.load(path + 'MIFGSM_' + atk_set + '_' + name_transformer + '_accuracy.pth')
    pgd_acc_transformer = torch.load(path + 'PGD_' + atk_set + '_' + name_transformer + '_accuracy.pth')
    fgsm_robustness_transformer = torch.load(path + 'FGSM_' + atk_set + '_' + name_transformer + '_robustness.pth')
    bim_robustness_transformer = torch.load(path + 'BIM_' + atk_set + '_' + name_transformer + '_robustness.pth')
    mifgsm_robustness_transformer = torch.load(path + 'MIFGSM_' + atk_set + '_' + name_transformer + '_robustness.pth')
    pgd_robustness_transformer = torch.load(path + 'PGD_' + atk_set + '_' + name_transformer + '_robustness.pth')

    fig, axs = plt.subplots(2, 1, layout='constrained')

    axs[0].plot(eps_vec, fgsm_acc_ours, '-*b')
    axs[0].plot(eps_vec, fgsm_acc_transformer, '--*b')
    axs[0].plot(eps_vec, bim_acc_ours, '-*r')
    axs[0].plot(eps_vec, bim_acc_transformer, '--*r')
    axs[0].plot(eps_vec, mifgsm_acc_ours, '-*g')
    axs[0].plot(eps_vec, mifgsm_acc_transformer, '--*g')
    axs[0].plot(eps_vec, pgd_acc_ours, '-*c')
    axs[0].plot(eps_vec, pgd_acc_transformer, '--*c')
    axs[0].set_xlabel(r'$\varepsilon$')
    axs[0].set_ylabel('ACC')
    axs[0].grid(True)
    # axs[0].legend()

    fgsm_ours, = axs[1].plot(eps_vec, fgsm_robustness_ours, '-*b')
    fgsm_deepcnn, = axs[1].plot(eps_vec, fgsm_robustness_transformer, '--*b')
    bim_ours, = axs[1].plot(eps_vec, bim_robustness_ours, '-*r')
    bim_deepcnn, = axs[1].plot(eps_vec, bim_robustness_transformer, '--*r')
    mifgsm_ours, = axs[1].plot(eps_vec, mifgsm_robustness_ours, '-*g')
    mifgsm_deepcnn, = axs[1].plot(eps_vec, mifgsm_robustness_transformer, '--*g')
    pgd_ours, = axs[1].plot(eps_vec, pgd_robustness_ours, '-*c')
    pgd_deepcnn, = axs[1].plot(eps_vec, pgd_robustness_transformer, '--*c')
    axs[1].set_xlabel(r'$\varepsilon$')
    axs[1].set_ylabel(r'$\hat{\rho}_{adv}$')
    axs[1].grid(True)
    # axs[1].legend()

    fig.legend((fgsm_ours, fgsm_deepcnn, bim_ours, bim_deepcnn, mifgsm_ours, mifgsm_deepcnn, pgd_ours, pgd_deepcnn),
               ('FGSM - Ours', 'FGSM - Transformer', 'BIM - Ours', 'BIM - Transformer',
                'MIFGSM - Ours', 'MIFGSM - Transformer', 'PGD - Ours', 'PGD - Transformer'),
               loc=(0.15, 0.35), prop={'size': 10})
    path_save = os.getcwd() + '\\figures\\adversarial_performance_vs_eps_' + atk_set + '.pdf'
    plt.savefig(path_save, format='pdf', bbox_inches='tight')
    plt.show()


def plot_metrics_vs_eps(name, atk_name, eps_vec):
    """
    Plots accuracy, F1 score, average robustness and fooling rate vs epsilon of some adversarial attack,
    and saves the figure.

    Args:
        name: model name.
        atk_name: attack name.
        eps_vec: vector of epsilon values.
    """
    path = os.getcwd() + '\\scores\\'

    acc = torch.load(path + atk_name + '_' + name + '_accuracy.pth')
    f1 = torch.load(path + atk_name + '_' + name + '_f1.pth')
    fool = torch.load(path + atk_name + '_' + name + '_fooling.pth')
    robustness = torch.load(path + atk_name + '_' + name + '_robustness.pth')

    acc_lambda_1 = torch.load(path + atk_name + '_' + name + '_lambda_1_accuracy.pth')
    f1_lambda_1 = torch.load(path + atk_name + '_' + name + '_lambda_1_f1.pth')
    fool_lambda_1 = torch.load(path + atk_name + '_' + name + '_lambda_1_fooling.pth')
    robustness_lambda_1 = torch.load(path + atk_name + '_' + name + '_lambda_1_robustness.pth')

    acc_lambda_2 = torch.load(path + atk_name + '_' + name + '_lambda_2_accuracy.pth')
    f1_lambda_2 = torch.load(path + atk_name + '_' + name + '_lambda_2_f1.pth')
    fool_lambda_2 = torch.load(path + atk_name + '_' + name + '_lambda_2_fooling.pth')
    robustness_lambda_2 = torch.load(path + atk_name + '_' + name + '_lambda_2_robustness.pth')

    acc_lambda_3 = torch.load(path + atk_name + '_' + name + '_lambda_3_accuracy.pth')
    f1_lambda_3 = torch.load(path + atk_name + '_' + name + '_lambda_3_f1.pth')
    fool_lambda_3 = torch.load(path + atk_name + '_' + name + '_lambda_3_fooling.pth')
    robustness_lambda_3 = torch.load(path + atk_name + '_' + name + '_lambda_3_robustness.pth')

    acc_lambda_4 = torch.load(path + atk_name + '_' + name + '_lambda_4_accuracy.pth')
    f1_lambda_4 = torch.load(path + atk_name + '_' + name + '_lambda_4_f1.pth')
    fool_lambda_4 = torch.load(path + atk_name + '_' + name + '_lambda_4_fooling.pth')
    robustness_lambda_4 = torch.load(path + atk_name + '_' + name + '_lambda_4_robustness.pth')

    acc_lambda_5 = torch.load(path + atk_name + '_' + name + '_lambda_5_accuracy.pth')
    f1_lambda_5 = torch.load(path + atk_name + '_' + name + '_lambda_5_f1.pth')
    fool_lambda_5 = torch.load(path + atk_name + '_' + name + '_lambda_5_fooling.pth')
    robustness_lambda_5 = torch.load(path + atk_name + '_' + name + '_lambda_5_robustness.pth')

    acc_deepcnn = torch.load(path + atk_name + '_DeepCNN_accuracy.pth')
    f1_deepcnn = torch.load(path + atk_name + '_DeepCNN_f1.pth')
    fool_deepcnn = torch.load(path + atk_name + '_DeepCNN_fooling.pth')
    robustness_deepcnn = torch.load(path + atk_name + '_DeepCNN_robustness.pth')

    fig, axs = plt.subplots(2, 2, layout='constrained')

    axs[0, 0].plot(eps_vec, acc, '-*k', label=name)
    axs[0, 0].plot(eps_vec, acc_lambda_1, '-*b', label=r'$\lambda=0.1$')
    axs[0, 0].plot(eps_vec, acc_lambda_2, '-*r', label=r'$\lambda=0.2$')
    axs[0, 0].plot(eps_vec, acc_lambda_3, '-*g', label=r'$\lambda=0.3$')
    axs[0, 0].plot(eps_vec, acc_lambda_4, '-*c', label=r'$\lambda=0.4$')
    axs[0, 0].plot(eps_vec, acc_lambda_5, '-*m', label=r'$\lambda=0.5$')
    axs[0, 0].plot(eps_vec, acc_deepcnn, '-*y', label='DeepCNN')
    axs[0, 0].set_xlabel(r'$\varepsilon$')
    axs[0, 0].set_ylabel('ACC')
    axs[0, 0].grid(True)
    # axs[0, 0].legend()

    axs[0, 1].plot(eps_vec, f1, '-*k', label=name)
    axs[0, 1].plot(eps_vec, f1_lambda_1, '-*b', label=r'$\lambda=0.1$')
    axs[0, 1].plot(eps_vec, f1_lambda_2, '-*r', label=r'$\lambda=0.2$')
    axs[0, 1].plot(eps_vec, f1_lambda_3, '-*g', label=r'$\lambda=0.3$')
    axs[0, 1].plot(eps_vec, f1_lambda_4, '-*c', label=r'$\lambda=0.4$')
    axs[0, 1].plot(eps_vec, f1_lambda_5, '-*m', label=r'$\lambda=0.5$')
    axs[0, 1].plot(eps_vec, f1_deepcnn, '-*y', label='DeepCNN')
    axs[0, 1].set_xlabel(r'$\varepsilon$')
    axs[0, 1].set_ylabel('F1')
    axs[0, 1].grid(True)
    # axs[0, 1].legend()
    # axs[0, 1].set_axis_off()

    axs[1, 0].plot(eps_vec, fool, '-*k', label=name)
    axs[1, 0].plot(eps_vec, fool_lambda_1, '-*b', label=r'$\lambda=0.1$')
    axs[1, 0].plot(eps_vec, fool_lambda_2, '-*r', label=r'$\lambda=0.2$')
    axs[1, 0].plot(eps_vec, fool_lambda_3, '-*g', label=r'$\lambda=0.3$')
    axs[1, 0].plot(eps_vec, fool_lambda_4, '-*c', label=r'$\lambda=0.4$')
    axs[1, 0].plot(eps_vec, fool_lambda_5, '-*m', label=r'$\lambda=0.5$')
    axs[1, 0].plot(eps_vec, fool_deepcnn, '-*y', label='DeepCNN')
    axs[1, 0].set_xlabel(r'$\varepsilon$')
    axs[1, 0].set_ylabel('FR')
    axs[1, 0].grid(True)
    # axs[1, 0].legend()

    model, = axs[1, 1].plot(eps_vec, robustness, '-*k', label='Original')
    model_1, = axs[1, 1].plot(eps_vec, robustness_lambda_1, '-*b', label=r'Ours with $\lambda=0.1$')
    model_2, = axs[1, 1].plot(eps_vec, robustness_lambda_2, '-*r', label=r'Ours with $\lambda=0.2$')
    model_3, = axs[1, 1].plot(eps_vec, robustness_lambda_3, '-*g', label=r'Ours with $\lambda=0.3$')
    model_4, = axs[1, 1].plot(eps_vec, robustness_lambda_4, '-*c', label=r'Ours with $\lambda=0.4$')
    model_5, = axs[1, 1].plot(eps_vec, robustness_lambda_5, '-*m', label=r'Ours with $\lambda=0.5$')
    deepcnn, = axs[1, 1].plot(eps_vec, robustness_deepcnn, '-*y', label='DeepCNN')
    axs[1, 1].set_xlabel(r'$\varepsilon$')
    axs[1, 1].set_ylabel(r'$\hat{\rho}_{adv}$')
    axs[1, 1].grid(True)
    # axs[1, 1].legend()

    fig.legend((model, model_1, model_2, model_3, model_4, model_5, deepcnn),
               ('Transformer', r'$\lambda=0.1$', r'$\lambda=0.2$',
                r'$\lambda=0.3$', r'$\lambda=0.4$', r'$\lambda=0.5$', 'DeepCNN'),
               loc=(0.6, 0.35), prop={'size': 10})
    path_save = os.getcwd() + '\\figures\\adversarial_performance_vs_eps_' + atk_name + '.pdf'
    plt.savefig(path_save, format='pdf', bbox_inches='tight')
    plt.show()


def plot_pqd_with_relevance_map(x, attn, y, yhat, scaler, le, vmax=None):
    """
    Plots PQD signal with the relevance map calculated by attention rollout.

    Args:
        x: PQ signal.
        attn: relevance map.
        y: true label.
        yhat: predicted label.
        scaler: used for inverse transform the PQ signal to the original scale.
        le: label encoding transformation function.
        vmax: used for scaling the relevance map.
    """
    x_np = x.cpu().numpy()
    x_np = scaler.inverse_transform(x_np).squeeze()
    attn_np = attn.cpu().numpy()
    len = np.size(x_np)
    t = np.linspace(0, 10 / 50, len) * 1000
    extent = [t[0] - (t[1] - t[0]) / 2., t[-1] + (t[1] - t[0]) / 2., 0.5, 1.5]
    plt.imshow(attn_np, cmap='coolwarm', aspect='auto', extent=extent, vmin=0,
               vmax=attn_np.max() if vmax is None else vmax)
    label = le.inverse_transform([y])[0]
    plt.plot(t, x_np, color='black')
    plt.colorbar(label='Attention Rollout')
    plt.xlabel('Time [ms]')
    plt.ylabel('Voltage [pu]')
    plt.title(f'True: {le.inverse_transform([y])[0]}, '
              f'Prediction: {le.inverse_transform([yhat])[0]}')
    path_save = os.getcwd() + '\\figures\\pqd_with_rollout_plot'
    path_save += '_regular_' if vmax is None else '_lambda_5_'
    plt.savefig(path_save + label + '.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    return attn_np.max()


def plot_pqd_natural_and_adversarial(x, x_adv, y, yhat, yhat_adv, scaler, le, name):
    """
    Plots PQD signal (natural and adversarial).

    Args:
        x: PQ signal.
        x_adv: adversarial PQ signal.
        y: true label.
        yhat: predicted label.
        yhat_adv: adversarial predicted label.
        scaler: for scaling back to p.u.
        le: label encoding transformation function.
        name: name of the model.
    """
    x = x.cpu().numpy()
    x = scaler.inverse_transform(x).squeeze()
    x_adv = x_adv.cpu().numpy()
    x_adv = scaler.inverse_transform(x_adv).squeeze()
    length = np.size(x)
    t = np.linspace(0, 10 / 50, length) * 1000  # 10 cycles, nominal frequency 50 Hz, used time units is ms
    true_label = le.inverse_transform([y])[0]
    predicted_label = le.inverse_transform([yhat])[0]
    predicted_label_adv = le.inverse_transform([yhat_adv])[0]
    plt.plot(t, x, '-.b')
    plt.plot(t, x_adv, ':r', linewidth=2)
    plt.legend(['Natural', 'Adversarial'], loc='lower left', fontsize=12)
    plt.ylim((-1.5, 1.5))
    plt.xlabel('Time [ms]')
    plt.ylabel('Voltage [pu]')
    plt.figtext(0.31, 0.92, predicted_label + ' ', fontsize=16, color='blue', ha='center')
    plt.figtext(0.69, 0.92, predicted_label_adv + ' ', fontsize=16, color='red', ha='center')
    plt.grid()
    path_save = os.getcwd() + '\\figures\\pqd_plot_natural_adversarial_' + name
    plt.savefig(path_save + '_' + true_label + '.pdf', format='pdf', bbox_inches='tight')
    plt.show()


def plot_pqd_with_attn(x, attn, y, yhat, scaler, le):
    x_np = x.cpu().numpy()
    x_np = scaler.inverse_transform(x_np).squeeze()
    time = np.linspace(0, 10 / 50, np.size(x_np))
    attn_np = torch.mean(attn, dim=0).squeeze().cpu().numpy()
    cmap = plt.get_cmap('coolwarm')
    norm = plt.Normalize(attn_np.min(), attn_np.max())

    plt.figure(figsize=(6, 4))
    plt.title(f'True: {le.inverse_transform(y.unsqueeze(0))[0]}, '
              f'Predicted: {le.inverse_transform(yhat.unsqueeze(0))[0]}')

    for i in range(len(x_np) - 1):
        plt.plot([time[i], time[i + 1]], [x_np[i], x_np[i + 1]], color=cmap(norm(attn_np[i])), linewidth=1)
    # axs.scatter(time, x_np, s=5, c=attn_np, cmap=cmap, norm=None, linewidths=5)

    plt.xlabel('Time [ms]')
    plt.ylabel('Voltage [p.u.]')
    plt.grid()
    # axs.legend()

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(attn_np)
    plt.colorbar(sm, label=r'$\bar{\mathcal{A}}$')
    plt.tight_layout()
    plt.show()


def plot_pqd(x, y, scaler, le):
    """
    Plots a PQD

    Args:
        x: PQ signal.
        y: true label.
        scaler: for scaling back to p.u.
        le: label encoding transformation function.
    """
    x = x.cpu().numpy()
    x = scaler.inverse_transform(x).squeeze()
    length = np.size(x)
    t = np.linspace(0, 10 / 50, length) * 1000
    label = le.inverse_transform([y])[0]
    plt.plot(t, x, color='blue')
    plt.xlabel('Time [ms]')
    plt.ylabel('Voltage [pu]')
    plt.ylim((-1.5, 1.5))
    # plt.title(label)
    plt.grid()
    path_save = os.getcwd() + '\\figures\\pqd_plot'
    plt.savefig(path_save + '_' + label + '.pdf', format='pdf', bbox_inches='tight')
    plt.show()
