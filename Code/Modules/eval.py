#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import pandas as pd
import torch
import torchio as tio


from .common_utils import *

class Aggregator:
    def __init__(self, grid_sampler):
        self.aggregator = tio.inference.GridAggregator(grid_sampler)

    def add_batch(self, outputs, locations):
        self.aggregator.add_batch(outputs, locations)

    def get_output_tensor(self):
        return self.aggregator.get_output_tensor()


class ModelEvaluator:
    def __init__(self, model, test_samplers_loader, batch_size, device, output_dir, logger=None):
        self.model = model
        self.test_samplers_loader = test_samplers_loader
        self.batch_size = batch_size
        self.device = device
        self.output_dir = output_dir
        self.logger = logger  # Adding the logger here

    def load_checkpoint_and_print_epoch(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint.get('epoch', 'Not available')
        print(
            f"Checkpoint at '{checkpoint_path}' was created at epoch: {epoch}")
        return checkpoint

    def save_image(self, image, file_name):
        os.makedirs(self.output_dir, exist_ok=True)
    
        if image.dim() == 3:
            image = image.unsqueeze(0)  
    
        image = tio.ScalarImage(tensor=image)

        output_file_path = os.path.join(self.output_dir, file_name)

        image.save(output_file_path)

    def evaluate(self):

        self.model.eval()  

        total_dice, total_iou, total_accuracy, total_specificity = 0, 0, 0, 0
        num_patches = 0

        with torch.no_grad():

            test_aggregators = []
            image_aggregators = []
            gt_aggregators = []

            for i, sampler in enumerate(self.test_samplers_loader):

                patch_loader = torch.utils.data.DataLoader(
                    sampler, batch_size=self.batch_size)
                aggregator_name = i

                test_aggregator = Aggregator(sampler)
                image_aggregator = Aggregator(sampler)
                gt_aggregator = Aggregator(sampler)

                test_aggregators.append(test_aggregator)
                image_aggregators.append(image_aggregator)
                gt_aggregators.append(gt_aggregator)

                for patches_batch in patch_loader:

                    images = patches_batch['image']['data'].float().to(
                        self.device)
                    masks = patches_batch['label']['data'].float().to(
                        self.device)
                    outputs = self.model(images)
                    locations = patches_batch['location']

                    image_aggregators[i].add_batch(images, locations)

                    gt_aggregators[i].add_batch(masks, locations)

                    probabilities = torch.sigmoid(outputs)

                    predicted_masks = (probabilities > 0.5).float()
                    
                    test_aggregators[i].add_batch(predicted_masks, locations)
              
                    dice, iou = calculate_dice_iou(
                        predicted_masks.cpu(), masks.cpu())
                    accuracy = compute_accuracy(
                        predicted_masks.cpu(), masks.cpu())
                    specificity = compute_specificity(
                        predicted_masks.cpu(), masks.cpu())

                    # Aggregate the metrics
                    total_dice += dice
                    total_iou += iou
                    total_accuracy += accuracy
                    total_specificity += specificity
                    num_patches += 1

# =============================================================================
#         # Get aggregated output
#         aggregated_masks_list = []
# 
#         # Get aggregated ouput tensors for each test image
#         for aggregator in self.aggregators:
#             aggregated_masks_list.append(aggregator.get_output_tensor())
# =============================================================================

        # Calculate average metrics over all batches
        average_dice = total_dice / num_patches
        average_iou = total_iou / num_patches
        average_accuracy = total_accuracy / num_patches
        average_specificity = total_specificity / num_patches

        # Log the metrics if logger is available
        if self.logger:
            metrics = {
                'Dice': average_dice,
                'IoU': average_iou,
                'Accuracy': average_accuracy,
                'Specificity': average_specificity
            }
            # Assuming epoch=0 for evaluation
            self.logger.log_metrics(metrics, epoch=0, prefix='Test')

        print(
            f"Average Dice Score: {average_dice}, Average IoU: {average_iou}")
        print(
            f"Average Accuracy: {average_accuracy}, Average Specificity: {average_specificity}")

        for i, test_aggregator in enumerate(test_aggregators):
            aggregated_masks = test_aggregator.get_output_tensor()
            for j, mask in enumerate(aggregated_masks):
                file_name = f"aggregated_test_image_{i}_{j}.nii.gz"
                self.save_image(mask.cpu(), file_name)

        for i, image_aggregator in enumerate(image_aggregators):
            aggregated_images = image_aggregator.get_output_tensor()
            for j, mask in enumerate(aggregated_images):
                file_name = f"aggregated_image_{i}_{j}.nii.gz"
                self.save_image(mask.cpu(), file_name)

        for i, gt_aggregator in enumerate(gt_aggregators):
            aggregated_gts = gt_aggregator.get_output_tensor()
            for j, mask in enumerate(aggregated_gts):
                file_name = f"aggregated_gt_{i}_{j}.nii.gz"
                self.save_image(mask.cpu(), file_name)

        metrics_df = pd.DataFrame({
            'Metric': ['Dice', 'IoU', 'Accuracy', 'Specificity'],
            'Value': [average_dice, average_iou, average_accuracy, average_specificity]
        })

        grandparent_directory = os.path.dirname(
            os.path.dirname(self.output_dir))

        csv_filename = os.path.join(
            grandparent_directory, 'evaluation_metrics.csv')
        print("CSV Filename:", csv_filename)  

        metrics_df.to_csv(csv_filename, index=False)

        print(f"Metrics saved to {csv_filename}")
