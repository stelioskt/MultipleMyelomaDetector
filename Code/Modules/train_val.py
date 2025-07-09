import os
import time

# import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import LambdaLR
from .common_utils import calculate_dice_iou
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# ================= Utility Functions ================= #
def Dice_loss(pred, target, smooth = 1.):
    pred= torch.sigmoid(pred)
    intersection = (pred * target).sum()
    dice = 1 - (2.*intersection + smooth)/(pred.sum() + target.sum() + smooth)
    return dice

# ================= Loss Class ================= #

class CombinedDiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        dice = Dice_loss(inputs, targets)
        bce = self.bce(inputs, targets)
        combined_loss = dice + bce
        return combined_loss, dice, bce
    
# ================= Trainer Class ================= #

class Trainer:
    # Initialization and setup
    def __init__(self, model, criterion, optimizer, scheduler, device, checkpoint_dir, logger, patience, grad_accumulation_steps=1):
        # Model and Device Settings
        self.device = device
        self.model = model.to(self.device)
        # self.model.half() # Uncomment if you need to set the model to half precision

        # Training Components
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = GradScaler()  # For mixed precision

        # Training Control Parameters
        self.patience = patience
        self.grad_accumulation_steps = grad_accumulation_steps

        # Training State
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.start_epoch = 0
        self.early_stopping_triggered = False
        
        # Timing
        self.epoch_start_time = None
        self.epoch_end_time = None

        # Utilities and Management
        self.logger = logger
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)


    # Training methods
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        metrics = self.initialize_metrics()
        
        for i, batch in enumerate(dataloader):
            inputs, labels = self._prepare_batch(batch)
            
            outputs, loss, dice_loss, bce_loss, dice_score, iou_score = self.process_batch(inputs, labels)
            
            self.update_optimizer(i, len(dataloader))
            
        self.log_epoch_metrics(metrics, len(dataloader), epoch)
        torch.cuda.empty_cache()  # Explicitly release GPU memory

    def initialize_metrics(self):
        return {'Loss': 0, 'Dice Loss': 0, 'BCE Loss': 0, 'Dice Score': 0, 'IoU Score': 0}

    def process_batch(self, inputs, labels):
        with autocast():  # Enable automatic mixed precision
            outputs = self.model(inputs)
            loss, dice_loss, bce_loss = self.criterion(outputs, labels)
            self.scaler.scale(loss / self.grad_accumulation_steps).backward()
    
        dice_score, iou_score = calculate_dice_iou(torch.sigmoid(outputs) > 0.5, labels > 0.5)
        return outputs, loss, dice_loss, bce_loss, dice_score, iou_score

    def update_optimizer(self, batch_index, total_batches):
        if (batch_index + 1) % self.grad_accumulation_steps == 0 or batch_index + 1 == total_batches:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

    def update_metrics(self, metrics, loss, dice_loss, bce_loss, dice_score, iou_score):
        metrics['Loss'] += loss.item() / self.grad_accumulation_steps
        metrics['Dice Loss'] += dice_loss.item() / self.grad_accumulation_steps
        metrics['BCE Loss'] += bce_loss.item() / self.grad_accumulation_steps
        metrics['Dice Score'] += dice_score / self.grad_accumulation_steps
        metrics['IoU Score'] += iou_score / self.grad_accumulation_steps

    def log_epoch_metrics(self, metrics, num_batches, epoch):
        avg_metrics = {k: v / num_batches for k, v in metrics.items()}
        self.logger.log_metrics(avg_metrics, epoch, 'Training')

    def train(self, train_dataloader, val_dataloader, num_epochs):
        last_checkpoint_file = None 
        torch.cuda.empty_cache()
    
        for epoch in range(self.start_epoch, num_epochs):
            self.train_epoch(train_dataloader, epoch)
            val_loss = self.validate_epoch(val_dataloader, epoch)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best = True
                self.epochs_no_improve = 0
            else:
                is_best = False
                self.epochs_no_improve += 1
    
            last_checkpoint_file = self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler, epoch, self.best_val_loss, is_best
            )
                
            if self.epochs_no_improve >= self.patience:
                self.early_stopping_triggered = True
                print(f"Early stopping triggered at epoch {epoch + 1}")

                if last_checkpoint_file:
                    self.start_epoch, self.best_val_loss, loaded_scheduler = self.checkpoint_manager.load_checkpoint(
                        self.model, self.optimizer, self.scheduler, last_checkpoint_file
                    )
                    if loaded_scheduler is not None:
                        self.scheduler = loaded_scheduler
                break  
    
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            else: self.optimizer.step()
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {val_loss}")
                        
    # Validation methods
    def validate_epoch(self, dataloader, epoch):
        self.model.eval()
        total_loss = 0.0  
        metrics = {'Dice Loss': 0, 'BCE Loss': 0, 'Dice Score': 0, 'IoU Score': 0}  # Other metrics
        num_batches = len(dataloader)

        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = self._prepare_batch(batch)
                outputs = self.model(inputs)
                loss, dice_loss, bce_loss = self.criterion(outputs, labels)

                total_loss += loss.item()  # Update total loss
                metrics['Dice Loss'] += dice_loss.item()
                metrics['BCE Loss'] += bce_loss.item()
                dice_score, iou_score = calculate_dice_iou(torch.sigmoid(outputs) > 0.5, labels > 0.5)
                metrics['Dice Score'] += dice_score.item()
                metrics['IoU Score'] += iou_score.item()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0  
        avg_metrics = {k: v / num_batches for k, v in metrics.items()}  
        
        self.logger.log_metrics({'Average Validation Loss': avg_loss}, epoch, 'Validation')
        self.logger.log_metrics(avg_metrics, epoch, 'Validation')
        
        return avg_loss  
        
    def _prepare_batch(self, batch):
        inputs = batch['image']['data'].float().to(self.device)
        labels = batch['label']['data'].float().to(self.device)
        return inputs, labels
    
    def _log_images(self, inputs, outputs, labels, step):
        slice_index = outputs.shape[2] // 2  # Assuming outputs is a 4D tensor (N,C,H,W)
        inputs_slice = inputs[:, :, slice_index]
        outputs_slice = outputs[:, :, slice_index]
        labels_slice = labels[:, :, slice_index]
        self.logger.log_images('Inputs', inputs_slice, step)
        self.logger.log_images('Predictions', outputs_slice, step)
        self.logger.log_images('Ground Truth', labels_slice, step)


class CheckpointManager:
    def __init__(self, directory):
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)
        self.last_checkpoint_filename = os.path.join(self.directory, "last_checkpoint.pth")
        self.best_checkpoint_filename = os.path.join(self.directory, "best_model_checkpoint.pth")

    def save_checkpoint(self, model, optimizer, scheduler, epoch, best_val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }
        
        if scheduler is not None:
            checkpoint['scheduler'] = scheduler.state_dict()

        # Save the current checkpoint as the last checkpoint
        torch.save(checkpoint, self.last_checkpoint_filename)

        # If this is the best checkpoint so far, save it separately
        if is_best:
            torch.save(checkpoint, self.best_checkpoint_filename)

        return self.last_checkpoint_filename if not is_best else self.best_checkpoint_filename

    def load_checkpoint(self, model, optimizer, scheduler, filename):
        print("=> Loading checkpoint")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            checkpoint['scheduler'] = scheduler.state_dict()
            scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint.get('epoch', 0), checkpoint.get('best_val_loss', float('inf'))


        

class TensorBoardLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def log_metrics(self, metrics, epoch, prefix=''):
        for key, value in metrics.items():
            self.writer.add_scalar(f'{prefix} {key}', value, epoch)

    def log_images(self, tag, img_tensor, step):
        self.writer.add_images(tag, img_tensor, step, dataformats='NCHW')

    def close(self):
        self.writer.close()

