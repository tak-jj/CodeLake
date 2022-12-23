import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    since = time.time()
    best_metric = 0.0
    best_model = None
    metric_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        model.train()
        train_loss = []
        for inputs, targets in tqdm(iter(train_loader)):
            inputs = inputs.float().to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
        
        tr_loss = np.mean(train_loss)

        val_loss, val_metric = val_model(model, test_loader, criterion, device)
        metric_history.append(val_metric)

        print(f'Epoch {epoch}, Train Loss: {tr_loss:.4f}, Val Loss: {val_loss:.4f}, Metric: {val_metric:.4f}')

        if val_metric > best_metric:
            best_model = model
            best_metric = val_metric

    time_elapsed = time.time() - since
    print('Train complete in {:0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Metric: {:.4f}'.format(best_metric))

    return best_model, metric_history

def val_model(model, test_loader, criterion, device):
    model.eval()

    model_preds = []
    true_labels = []
    val_loss = []

    with torch.no_grad():
        for inputs, targets in tqdm(iter(test_loader)):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss.append(loss.item())

            model_preds += outputs.argmax(1).detach().cpu().numpy().tolist()
            true_labels += targets.detach().cpu().numpy().tolist()
    
    val_metric = metric(true_labels, model_preds)
    return np.mean(val_loss), val_metric
