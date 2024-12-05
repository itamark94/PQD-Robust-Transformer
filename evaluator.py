from time import time
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def evaluate_classification_performance(model, dataloader, device):
    """
    Evaluates performance of a PQD classifier.

    Args:
        model: PQD classifier.
        dataloader: inputs and labels.
        device: cuda or cpu.

    Returns:
        acc: accuracy score.
        f1: F1 score.
        cm: confusion matrix.
    """
    y_true = torch.tensor([], dtype=torch.long, device=device)
    y_pred = torch.tensor([], dtype=torch.long, device=device)
    start = time()
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            predictions = model(x)
            _, predicted = torch.max(predictions, dim=1)
            y_true = torch.cat((y_true, y))
            y_pred = torch.cat((y_pred, predicted))

    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    acc = accuracy_score(y_true_np, y_pred_np)
    f1 = f1_score(y_true_np, y_pred_np, average='macro')
    cm = confusion_matrix(y_true_np, y_pred_np, normalize=None)

    end = time()
    print(f"Inference time = {end - start:.0f} seconds")

    return acc, f1, cm


def average_robustness(inputs, adv_inputs):
    """
    Calculates the average robustness of adversarial inputs.

    Args:
        inputs: tensor in the shape of (batch_size, 1, input_length).
        adv_inputs: tensor in the shape of (batch_size, 1, input_length).

    Returns:
        avg_robustness: average robustness of the adversarial inputs.
    """
    perturbation_norm = torch.linalg.norm(adv_inputs - inputs, dim=2)
    inputs_norm = torch.linalg.vector_norm(inputs, dim=2)
    avg_robustness = torch.sum(perturbation_norm / inputs_norm) / len(inputs)

    return avg_robustness


def fooling_rate(model, inputs, inputs_adv, batch_size, device):
    """
    Calculates fooling rate of a model given inputs and their corresponding adversarial inputs.

    Args:
        model: PQD classifier.
        inputs: natural PQ signals.
        inputs_adv: adversarial PQ signals.
        batch_size: batch size.
        device: cuda or cpu.

    Returns:
        fr: fooling rate.
    """
    N = len(inputs)
    data_loader = DataLoader(TensorDataset(inputs, inputs_adv), batch_size, shuffle=False)
    total_fooled = 0

    model.eval()
    with torch.no_grad():
        for i, (inputs_batch, adv_inputs_batch) in enumerate(data_loader):
            inputs_batch, adv_inputs_batch = inputs_batch.to(device), adv_inputs_batch.to(device)
            _, predictions = torch.max(model(inputs_batch), dim=1)
            _, predictions_adv = torch.max(model(adv_inputs_batch), dim=1)
            total_fooled += torch.sum(predictions != predictions_adv).item()

    fr = total_fooled / N

    return fr
