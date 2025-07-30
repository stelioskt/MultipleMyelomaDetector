#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 14:13:18 2024

@author: georgeb
"""
import os
import sys
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from Modules import *


# Set up working directory and configuration
wd= '/home/georgeb/Desktop/code/trans_unet_framework' #in-script input
sys.path.append(wd)

# YAML configuration file
config = 'config.yml' #in-script input
config = yaml_config.load_config(os.path.join(wd, config))

# Extract configuration settings
dataset_config = config['dataset_config']
unetr_config = config['unetr_config']
training_config = config['training_config']
testing_config = config['testing_config']
logging_config = config['logging_config']

# Create the directories defined by the config 
common_utils.create_directories(config)

# Dataset setup
patched_nii_data_loader = data_prep.PatchedNiftiDatasetLoader(
    dataset_config['folder_path'],
    dataset_config['img_space'],
    dataset_config['img_size']
    )

# train_dataset, val_dataset, test_grid_samplers = patched_nii_data_loader.aggregated_train_val_plus_test_grid_sampler()
print("dataset created...")
# DataLoader setup
train_loader, val_loader, test_grid_samplers =  patched_nii_data_loader.create_dataloaders(
                                                    batch_size=dataset_config['batch_size'],
                                                    num_workers=dataset_config['num_workers'],
                                                    train_samples=dataset_config['train_samples'],
                                                    val_samples=dataset_config['val_samples'],
                                                    test_samples=dataset_config['test_samples'],
                                                    shuffle=dataset_config['shuffle'] 
                                                    )

print("dataloaders created...")

# =============================================================================
# def poly_decay(epoch, max_epochs, initial_lr):
#     decay_factor = (1 - epoch/ max_epochs ) ** 0.9
#     return decay_factor
# =============================================================================
# =============================================================================
# #These work with SGD and LambdaLR scheduler
# initial_lr = 0.01
# max_epochs = 400
# =============================================================================
# Model

my_model = unetr.UNETR(**unetr_config)
model_state_dict = my_model.state_dict()

# Loss function, optimizer, and scheduler setup
criterion = train_val.CombinedDiceBCELoss()

# optimizer = train_val.Optimizer(my_model.parameters(), ilr=training_config['learning_rate'], momentum=0.99, nesterov=True).sgd_optimizer()
# optimizer = optim.SGD(my_model.parameters(), lr=initial_lr, momentum=0.99, nesterov=True)
optimizer = optim.AdamW(my_model.parameters(), training_config['learning_rate'])#, weight_decay=1e-5)

# scheduler = train_val.Scheduler(optimizer, max_epochs=training_config['num_epochs']).LambdaLR()
# scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: poly_decay(epoch, max_epochs, initial_lr))
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Gradient scaler for mixed precision training
# scaler = train_val.GradScaler()

logger = train_val.TensorBoardLogger(log_dir=logging_config['log_dir'])
print("ready to proceed!")
trainer = train_val.Trainer(
    model=my_model, 
    criterion=criterion, 
    optimizer=optimizer, 
    scheduler=None, 
    device=torch.device(training_config["device"]), 
    logger=logger, 
    patience=2000, # Set patience for early stopping
    grad_accumulation_steps=1,
    checkpoint_dir=training_config['checkpoint_dir']
    )

# =============================================================================
# # Checkpointed training
# 
# # Create an instance of the CheckpointManager class, 
# checkpoint_manager = train_val.CheckpointManager(training_config['checkpoint_dir'])
# 
# # Provide the path to the checkpoint file
# latest_checkpoint = os.path.join(training_config['checkpoint_dir'],'last_checkpoint.pth')
# 
# # Load the checkpoint using the Checkpoint Manager
# start_epoch, best_val_loss = checkpoint_manager.load_checkpoint(
#     trainer.model,
#     trainer.optimizer,
#     trainer.scheduler,
#     latest_checkpoint
#     )
# 
# # Set the start_epoch and best_val_loss values in the Trainer instance
# trainer.start_epoch = start_epoch
# trainer.best_val_loss = best_val_loss
# # Continue training
# =============================================================================

trainer.train(train_loader, val_loader, num_epochs=training_config['num_epochs'])

# Testing 
evaluator = eval.ModelEvaluator(
    trainer.model, 
    test_grid_samplers, 
    dataset_config['batch_size'],
    trainer.device, 
    testing_config['output_dir'],
    logger  # Pass the logger instance here
    )
evaluator.evaluate()

# Close the logger
logger.close()


