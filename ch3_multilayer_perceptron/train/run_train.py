# ch3_multilayer_perceptron/train/run_train.py

def training_loop(n_epochs, optimizer, model, loss_fn, 
                  x_train, 
                  y_train, 
                 ):
    for epoch in range(1, n_epochs + 1):
        y_p_train = model(x_train)
        loss_train = loss_fn(y_p_train, y_train)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train.item():.4f}")