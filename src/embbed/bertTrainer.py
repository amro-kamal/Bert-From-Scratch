from bert import bert
import torch


class bertTrainer:

    def Train(model,device, criterion, data_loader, val_loader , optimizer, num_epochs ,scheduler=None ,scheduler2=None , model_path='model.ckpt'):
        """Simple training loop for a PyTorch model.""" 
        best_val_acc=0
        # Make sure model is in training mode.
        model.train()
        optimizer.zero_grad()

        # Move model to the device (CPU or GPU).
        model.to(device)
        
        # Exponential moving average of the loss.
        ema_loss = None
        
        print('----- Training -----')
        # Loop over epochs.
        for epoch in range(num_epochs):
            total=0
            train_running_correct=0
              # Loop over data.
            for batch_idx, input in enumerate(data_loader):
                # Forward pass.
                CLS , preds_tokens = model(input['tokens'].to(device) , input['segments'])
                target=input['target'].to(device)
                loss = criterion(output.to(device), target)


                  # Backward pass.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            print('##'*20)
