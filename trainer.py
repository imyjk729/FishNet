import os
import torch

from tensorboardX import SummaryWriter


class Trainer():
    def __init__(self, model, optimizer, scheduler, loss_function, config) :
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.config = config
        self.writer = SummaryWriter()

        super().__init__()

        self.device = next(model.parameters()).device

    def train(self, train_loader) :
        # Turn train mode on.
        self.model.train()
        
        total_loss = 0.
        total_correct = 0.

        for step, (input, targets) in enumerate(train_loader):
            input = input.to(self.device)
            targets = targets.to(self.device)

            logits = self.model(input)
            _, preds = torch.max(logits, 1)
            total_correct += torch.sum(preds == targets)

            loss = self.loss_function(logits, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss * input.size(0)
            
        self.scheduler.step()
        # Train ACC and loss
        acc = total_correct / len(train_loader.dataset)
        loss_avg = total_loss / len(train_loader.dataset)

        return acc, loss_avg

    
    def validate(self, valid_loader):
        # Turn evaluation mode on.
        self.model.eval()  
        total_loss = 0.
        total_correct = 0.    

        with torch.no_grad():
            for step, (input, targets) in enumerate(valid_loader):
                input = input.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(input)
                _, preds = torch.max(logits, 1)
                total_correct += torch.sum(preds == targets)

                loss = self.loss_function(logits, targets)
                total_loss += loss * input.size(0)

        # Valid ACC and loss
        acc = total_correct / len(valid_loader.dataset)  
        loss_avg = total_loss / len(valid_loader.dataset)  

        return acc, loss_avg
    
    
    def run(self, train_loader, valid_loader):
        best_acc = -1

        for epoch in range(self.config.n_epochs):
            train_acc, train_loss = self.train(train_loader)
            valid_acc, valid_loss = self.validate(valid_loader)

            # Write down ACC and loss on Tensorboard
            self.writer.add_scalar('Train_loss', train_loss, epoch)
            self.writer.add_scalar('Valid_loss', valid_loss, epoch)
            self.writer.add_scalar('Train_ACC', train_acc, epoch)
            self.writer.add_scalar('Valid_ACC', valid_acc, epoch)

            print("Epoch(%d/%d): train_loss=%.4f  valid_loss=%.4f  train_acc=%.4f  valid_acc=%.4f" % (
                epoch + 1,
                self.config.n_epochs,
                train_loss,
                valid_loss,
                train_acc,
                valid_acc,
            ))
            
            # Save the best model.
            if valid_acc > best_acc:
                best_acc = valid_acc

                if self.config.out:
                    if not os.path.exists(self.config.MODEL_PATH):
                        os.mkdir(self.config.MODEL_PATH)

                    output_path = os.path.join(self.config.MODEL_PATH, self.config.MODEL)
                    torch.save(self.model, output_path)

        self.writer.close()