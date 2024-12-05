import os
import argparse
import torch
from sklearn.preprocessing import StandardScaler

from utils import make_folders, load_dataset_classes, split_train_test, split_train_val
from utils_plot import plot_metrics_vs_eps, plot_metrics_vs_eps_diff_atks
from train_pqd_classifier import train_pqd_classifier_transformer, train_pqd_classifier_cnn
from evaluate_pqd_classifier import evaluate_pqd_classifier
from evaluate_attacked_pqd_classifier import evaluate_white_box_attack, evaluate_black_box_attack
from plot_pqd import plot_pqd_with_relevance_map_main, plot_pqds_natural, plot_pqds_natural_adversarial


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--val_split', type=float, default=0.25)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--batch_size_test', type=int, default=32)
    parser.add_argument('--batch_size_plot', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--epochs_exp', type=int, default=10)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--patience_exp', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_exp', type=float, default=1e-4)
    parser.add_argument('--lambda_class', default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--dim_feedforward', type=int, default=128)
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--dropout_attn', type=float, default=0.25)
    parser.add_argument('--exp', type=bool, default=False)
    parser.add_argument('--eps', type=float, default=0.06)
    parser.add_argument('--eps_vec', default=[0.01, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2])
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--decay', type=float, default=0.9)

    return parser.parse_args()


if __name__ == '__main__':
    make_folders()
    args = parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    labels_order = ['Normal', 'Sag', 'Swell', 'Interruption', 'Harmonics', 'Notch', 'Flicker',
                    'Spike', 'Impulsive transient', 'Oscillatory transient']
    labels_indices = [118, 18, 384, 2, 5, 13, 15, 7, 21, 146]  # Selected PQDs from the test-set for plots

    n_classes = len(labels_order)
    x, y, gt, le = load_dataset_classes(os.getcwd() + '\\datasets\\', labels_order)
    x_train, y_train, gt_train, x_test, y_test, gt_test = split_train_test(x, y, gt, args.train_split)
    x_train, y_train, gt_train, x_val, y_val, gt_val = split_train_val(x_train, y_train, gt_train,
                                                                       args.val_split, args.seed)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    names = ['DeepCNN', 'Transformer']
    atk_names = ['PGD']
    """
    Change the lists 'names' and 'atk_names' according to selected architectures and adversarial attacks.
    For example: names = ['Transformer'], atk_names = ['PGD']
    """

    label = 'Spike'  # used for plotting natural and adversarial PQDs and their relevance maps.

    train_pqd_classifier_transformer(x_train, y_train, gt_train, x_val, y_val, gt_val, n_classes, args, device)
    train_pqd_classifier_cnn(x_train, y_train, gt_train, x_val, y_val, gt_val, n_classes, args, device)
    """
    In case you wish to evaluate PQD classification performance of the models from the original paper and plot results, 
    make sure you load the models from the folder 'models_paper'. Otherwise, you can train the models which will be 
    saved in the folder 'models' and use them instead. 
    """
    evaluate_pqd_classifier(names[0], x_test, y_test, gt_test, n_classes, le, device, args)
    for name in names:
        for atk_name in atk_names:
            evaluate_white_box_attack(name, atk_name, x_test, y_test, n_classes, le, device, args)
            evaluate_black_box_attack(name, atk_name, x_test, y_test, n_classes, le, device, args)

    plot_metrics_vs_eps(name='Transformer', atk_name='PGD_black_box', eps_vec=args.eps_vec)
    plot_metrics_vs_eps_diff_atks(atk_set='black_box', eps_vec=args.eps_vec)
    plot_pqds_natural(x_test, y_test, le, scaler, labels_indices)
    plot_pqds_natural_adversarial('Transformer', x_test, y_test, le, scaler, n_classes, labels_indices, device, args)
    plot_pqd_with_relevance_map_main(label, x_test, y_test, n_classes, le, scaler, device, args)
