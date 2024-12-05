import torch


def attack(atk, data_loader, device):
    """
    Performs adversarial attack on a PQD classifier.

    Args:
        atk: attack method, already includes the target model.
        data_loader: inputs and labels.
        device: cpu or cuda.

    Returns:
        x_adv: adversarial inputs.
        y: true labels.
    """
    x_adv = []
    y = []

    for x_batch, y_batch in data_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_batch_adv = [atk(x_batch, y_batch)]
        x_adv += x_batch_adv
        y += [y_batch]

    x_adv = torch.cat(x_adv).to("cpu")
    y = torch.cat(y).to("cpu")

    return x_adv, y
