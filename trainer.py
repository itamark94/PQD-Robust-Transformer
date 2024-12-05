import os
import torch
from time import time


def train(model, name, train_loader, val_loader, optimizer, criteria, args, lambda_class):
    """
    Trains a PQD classifier and save the model and vectors of loss and accuracy of training and validation.

    Args:
        model: PQD classifier.
        name: string which indicates the architecture of the classifier.
        train_loader: training dataset loader.
        val_loader: validation dataset loader.
        optimizer: optimization algorithm.
        criteria: loss functions.
        args: arguments (training parameters).
        lambda_class: size of lambda.
    """
    train_loss_vec, val_loss_vec = [], []
    train_acc_vec, val_acc_vec = [], []
    best_loss = torch.inf
    best_acc = 0.0
    bad_epochs = 0
    if args.exp:
        print(f"Finetuning procedure with lambda = {lambda_class:.1f}")
        path_model = "models\\" + name + "_model.pth"
        model.load_state_dict(torch.load(os.path.join(os.getcwd(), path_model)))
        name += "_lambda_" + str(int(lambda_class*10))
        epochs = args.epochs_exp
        patience = args.patience_exp
    else:
        print("Training " + name + " for classification")
        epochs = args.epochs
        patience = args.patience
    path_model = "models\\" + name + "" + "_model.pth"
    path_train_loss = "losses\\" + name + "" + "_train_loss.pth"
    path_val_loss = "losses\\" + name + "" + "_val_loss.pth"
    path_train_acc = "accuracies\\" + name + "" + "_train_acc.pth"
    path_val_acc = "accuracies\\" + name + "" + "_val_acc.pth"
    device = next(model.parameters()).device

    start = time()
    for epoch in range(epochs):
        if args.exp:
            train_loss, train_acc = train_step_exp(model, train_loader, optimizer, criteria, device, lambda_class)
            val_loss, val_acc = val_step_exp(model, val_loader, criteria, device, lambda_class)
        else:
            train_loss, train_acc = train_step_class(model, train_loader, optimizer, criteria[0], device)
            val_loss, val_acc = val_step_class(model, val_loader, criteria[0], device)
        train_loss_vec.append(train_loss)
        train_acc_vec.append(train_acc)
        val_loss_vec.append(val_loss)
        val_acc_vec.append(val_acc)
        if val_loss < best_loss:
            print("--------------------------------------------Saving good model--------------------------------------------")
            best_loss = val_loss
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(os.getcwd(), path_model))
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs == patience:
                print("Early stopping")
                break
        print(f"Epoch={epoch + 1} | Bad epochs={bad_epochs} | "
              f"Training: ACC={train_acc * 100:.2f}%, Loss={train_loss:.8f} | "
              f"Validation: ACC={val_acc * 100:.2f}%, Loss={val_loss:.8f}")
    end = time()
    print(f"Training time = {end - start:.0f} seconds")
    print(f"Best ACC = {best_acc * 100:.2f} | Best Loss = {best_loss}")
    torch.save(train_loss_vec, os.path.join(os.getcwd(), path_train_loss))
    torch.save(train_acc_vec, os.path.join(os.getcwd(), path_train_acc))
    torch.save(val_loss_vec, os.path.join(os.getcwd(), path_val_loss))
    torch.save(val_acc_vec, os.path.join(os.getcwd(), path_val_acc))


def train_step_class(model, dataloader, optimizer, criterion, device):
    """
    Performs a regular training step (one epoch) and returns accuracy and loss.

    Args:
        model: PQD classifier (Transformer without the relevance fine-tuning procedure or DeepCNN).
        dataloader: training dataset loader.
        optimizer: optimization algorithm.
        criterion: loss function (cross-entropy).
        device: cuda or cpu.

    Returns:
        loss: loss.
        acc: accuracy.
    """
    model.train()
    loss = 0.0
    correct = 0
    total = 0

    for x, y, _ in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        predictions = model(x)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        loss += loss.data.item()
        _, predicted = predictions.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

    loss /= len(dataloader)
    acc = correct / total

    return loss, acc


def val_step_class(model, dataloader, criterion, device):
    """
    Performs a regular validation step (one epoch) and calculates loss and accuracy.

    Args:
        model: PQD classifier (Transformer without the relevance fine-tuning procedure or DeepCNN).
        dataloader: validation dataset loader.
        criterion: loss function (cross-entropy).
        device: cuda or cpu.

    Returns:
        loss: loss.
        acc: accuracy.
    """
    model.eval()
    loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y, _ in dataloader:
            x, y = x.to(device), y.to(device)
            predictions = model(x)
            loss = criterion(predictions, y)
            loss += loss.item()
            _, predicted = predictions.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    loss /= len(dataloader)
    acc = correct / total

    return loss, acc


def train_step_exp(model, dataloader, optimizer, criteria, device, lambda_class):
    """
    Performs a training step of the relevance fine-tuning procedure for the Transformer (one epoch) and calculates loss
    (weighted sum of classification loss and relevance loss) and accuracy.

    Args:
        model: Transformer PQD classifier.
        dataloader: training dataset loader.
        optimizer: optimization algorithm.
        criteria: list of loss functions (first for classification, second for relevance).
        device: cuda or cpu.
        lambda_class: hyperparameter for weighting the classification loss out of the total loss.

    Returns:
        loss: loss.
        acc: accuracy.
    """
    model.train()
    loss = 0.0
    correct = 0
    total = 0

    for x, y, gt in dataloader:
        x, y, gt = x.to(device), y.to(device), gt.to(device)
        optimizer.zero_grad()
        predictions, rollout = model.forward_and_attention_rollout(x)
        loss_class = criteria[0](predictions, y)
        rollout = rollout.mean(dim=1)
        loss_exp = criteria[1](rollout, gt)
        loss = lambda_class * loss_class + (1-lambda_class) * loss_exp
        loss.backward()
        optimizer.step()
        loss += loss.data.item()
        _, predicted = predictions.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

    loss /= len(dataloader)
    acc = correct / total

    return loss, acc


def val_step_exp(model, dataloader, criteria, device, lambda_class):
    """
    Performs a validation step of the relevance fine-tuning procedure for the Transformer (one epoch) and calculates
    loss (weighted sum of classification loss and relevance loss) and accuracy.

    Args:
        model: Transformer PQD classifier.
        dataloader: training dataset loader.
        criteria: list of loss functions (first for classification, second for relevance).
        device: cuda or cpu.
        lambda_class: hyperparameter for weighting the classification loss out of the total loss.

    Returns:
        loss: loss.
        acc: accuracy.
    """
    model.eval()
    loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y, gt in dataloader:
            x, y, gt = x.to(device), y.to(device), gt.to(device)
            predictions, rollout = model.forward_and_attention_rollout(x)
            loss_class = criteria[0](predictions, y)
            rollout = rollout.mean(dim=1)
            loss_exp = criteria[1](rollout, gt)
            loss = lambda_class * loss_class + (1 - lambda_class) * loss_exp
            loss += loss.item()
            _, predicted = predictions.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    loss /= len(dataloader)
    acc = correct / total

    return loss, acc
