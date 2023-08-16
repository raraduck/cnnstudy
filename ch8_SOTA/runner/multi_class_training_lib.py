import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

def training_3min(model, train_loader, loss_fn, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        # loss = F.cross_entropy(output, target)
        loss = loss_fn(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    train_loss = running_loss / len(train_loader)
    return train_loss

def evaluate_3min(model, test_loader, loss_fn, device):
    model.eval()
    running_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 배치 오차를 합산
            # running_loss += F.cross_entropy(output, target, reduction='sum').item()
            running_loss += loss_fn(output, target).item()
            
            # 가장 높은 값을 가진 인덱스가 바로 예측값
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def mclass_training_loop_3min(n_epochs, model, loss_fn, optimizer, scheduler, train_loader, test_loader, saver, device='cpu'):
    train_losses = []
    for epoch in range(1, n_epochs + 1):
        scheduler.step()
        train_loss = training_3min(model, train_loader, loss_fn, optimizer, epoch, device)
        train_losses.append(train_loss)
        test_loss, test_accuracy = evaluate_3min(model, test_loader, loss_fn, device)
        print(f'[{epoch}] Test Loss: {test_loss}, Accuracy: {test_accuracy}%')
        if epoch > 10:
            saver.save_at_best_test_loss(epoch, test_loss)
        
        
        
        
        
def training_basic(model, train_loader, loss_fn, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_fn(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    train_loss = running_loss / len(train_loader)
    return train_loss

    
def evaluate_basic(model, test_loader, loss_fn, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            
            running_loss += loss_fn(outputs, labels).item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    test_loss = running_loss / len(test_loader)
    test_accuracy = 100. * correct / total
    return test_loss, test_accuracy
    
    
def mclass_training_loop_basic(n_epochs, model, loss_fn, optimizer, scheduler, train_loader, test_loader, saver, device='cpu'):
    train_losses = []
    for epoch in range(1, n_epochs + 1):
        scheduler.step()
        train_loss = training_basic(model, train_loader, loss_fn, optimizer, epoch, device)
        train_losses.append(train_loss)
        test_loss, test_accuracy = evaluate_basic(model, test_loader, loss_fn, device)
        print(f'[{epoch}] Test Loss: {test_loss}, Accuracy: {test_accuracy}%')
        if epoch > 10:
            saver.save_at_best_test_loss(epoch, test_loss)