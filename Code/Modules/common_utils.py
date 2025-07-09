#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

def calculate_dice_iou(outputs, labels):
    outputs = outputs > 0.5
    labels = labels > 0.5

    outputs_flat = outputs.view(-1)
    labels_flat = labels.view(-1)

    intersection = (outputs_flat & labels_flat).sum()
    total = (outputs_flat | labels_flat).sum()
    union = total - intersection
    TP = intersection
    FP = (outputs_flat & ~labels_flat).sum()
    FN = (~outputs_flat & labels_flat).sum()

    dice = (2 * TP).float() / (2 * TP + FP + FN).float()

    iou = TP.float() / (TP + FP + FN).float()

    return dice, iou

def compute_accuracy(outputs, labels):
    predictions = outputs > 0.5
    labels = labels > 0.5

    TP = (predictions & labels).sum()
    TN = (~predictions & ~labels).sum()
    total = predictions.numel()  

    accuracy = (TP + TN).float() / total

    return accuracy.item()

def compute_specificity(outputs, labels):
    predictions = outputs > 0.5
    labels = labels > 0.5

    TN = (~predictions & ~labels).sum()
    FP = (predictions & ~labels).sum()

    specificity = TN.float() / (TN + FP).float() if (TN + FP) > 0 else 0.0  # Handle division by zero

    return specificity.item()

def create_directories(config):
    checkpoint_dir = config['training_config']['checkpoint_dir']
    log_dir = config['logging_config']['log_dir']
    output_dir = config['testing_config']['output_dir']

    for directory in [checkpoint_dir, log_dir, output_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
