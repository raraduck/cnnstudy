# ch4_dataloader/utils/run_training.py
import torch

def training_loop(n_epochs, optimizer, model, loss_fn, 
                  train_loader, 
                  device='cpu'):
    losses = []
    for i in range(n_epochs):
        for j,[image, label] in enumerate(train_loader):
            x = image.to(device)
            y_ = label.to(device)
    
            output = model(x)
            loss = loss_fn(output, y_)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if j % 1000 == 0:
                print(loss)
                losses.append(loss.cpu().detach().numpy())
    return losses


def validate(model, train_loader, val_loader, device='cpu'):
    accdict = {}
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1) # <1>
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

        print("Accuracy {}: {:.2f}".format(name , correct / total))
        accdict[name] = correct / total
    return accdict
