import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

def training_3min(model, train_loader, loss_fn, optimizer, epoch, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.cross_entropy(output, target)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()


def evaluate_3min(model, test_loader, loss_fn, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 배치 오차를 합산
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            # test_loss += loss_fn(output, target, reduction='sum').item()
            
            # 가장 높은 값을 가진 인덱스가 바로 예측값
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def mclass_training_loop_3min(n_epochs, model, loss_fn, optimizer, scheduler, train_loader, test_loader, device='cpu'):
    EPOCHS=n_epochs
    for epoch in range(1, EPOCHS + 1):
        scheduler.step()
        training_3min(model, train_loader, loss_fn, optimizer, epoch, device)
        test_loss, test_accuracy = evaluate_3min(model, test_loader, loss_fn, device)

        print(f'[{epoch}] Test Loss: {test_loss}, Accuracy: {test_accuracy}%')
        
        

def mclass_training_loop_basic(n_epochs, model, loss_fn, optimizer, scheduler, train_loader, test_loader, device='cpu'):
    loss_ = []
    n = len(train_loader)
    for epoch in range(n_epochs):
        
        model.train()
        running_loss = 0.0
        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        loss_.append(running_loss / n)
        print('[%d] loss: %.3f' %(epoch + 1, running_loss /len(train_loader)))
    
    plt.plot(loss_)
    plt.title("Training Loss")
    plt.xlabel("epoch")
    plt.show()
