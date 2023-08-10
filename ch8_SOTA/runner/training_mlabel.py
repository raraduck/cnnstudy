# ch7_medmnist/runner/run_loop.py
import tqdm
import torch 
import numpy as np

# def validate(model, train_loader, val_loader, device='cpu'):
#     accdict = {}
#     for name, loader in [("train", train_loader), ("val", val_loader)]:
#         correct = 0
#         total = 0

#         with torch.no_grad():
#             for imgs, labels in loader:
#                 imgs = imgs.to(device=device)
#                 labels = labels.to(device=device)
#                 outputs = model(imgs)
#                 _, predicted = torch.max(outputs, dim=1) # <1>
#                 total += labels.shape[0]
#                 correct += int((predicted == labels).sum())

#         print("Accuracy {}: {:.2f}".format(name , correct / total))
#         accdict[name] = correct / total
#     return accdict


def mlabel_eval_net(model, data_loader, device="cpu"):
    model.eval()
    ys = []
    ypreds = []
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            y_pred = model(x)
        y_pred = y_pred > 0.5
        ys.append(y)
        ypreds.append(y_pred)
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    acc = (ys == ypreds).cpu().sum() / len(ys)
    return acc.item()


def mlabel_training_loop(n_epochs, optimizer, model, loss_fn, train_loader, test_loader, device='cpu'):
    train_losses = []
    train_acc = []
    val_acc = []
    for epoch in range(n_epochs):
        running_loss = 0.0
        model.train()
        n = 0
        n_acc = 0
        # 시간이 많이 걸리므로 tqdm을 사용해서 진행바를 표시
        for i, (xx, yy) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            xx = xx.to(device)
            yy = yy.to(device)
            h = model(xx)
            loss = loss_fn(h, yy)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n += len(xx)
            y_pred = h > 0.5
            cmp = (yy == y_pred).cpu()
            n_acc += cmp.sum().item()
        train_losses.append(running_loss / i)
        # 훈련 데이터의 예측 정확도
        train_acc.append(n_acc / n)

        # 검증 데이터의 예측 정확도
        val_acc.append(mlabel_eval_net(model, test_loader, device))
        print(epoch, train_losses[-1], train_acc[-1], val_acc[-1], flush=True)
    print('-----------training finished-----------')
    print('train_losses: ', train_losses)
    print('train_acc: ', train_acc)
    print('val_acc: ', val_acc)
