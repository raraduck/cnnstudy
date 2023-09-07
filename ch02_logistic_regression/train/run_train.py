# ch2_logistic_regression/train/run_train.py

def training_loop(n_epochs, optimizer, model, loss_fn, 
                  x_train, 
                  y_train, 
                 ):
    # losses = []
    for epoch in range(1, n_epochs + 1):
        y_p_train = model(x_train)
        loss_train = loss_fn(y_p_train, y_train)

        # y_p_val = model(x_val)
        # loss_val = loss_fn(y_p_val, y_val)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # losses.append(loss_train.item())
        if epoch == 1 or epoch % 100 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train.item():.4f}")
    # return losses