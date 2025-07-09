#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import math
import torch
import torch.nn as nn
import torchio as tio
import nibabel as nib

import SimpleITK as sitk

from torch.utils.data import Dataset, DataLoader, random_split
import torchio.transforms as T

class PatchedNiftiDatasetLoader(Dataset):
    def __init__(self, folder_path, img_space, img_size, indices=None):
        self.folder_path = folder_path
        self.img_space = tuple(img_space) 
        self.img_size = tuple(img_size)
        self.indices = indices
        self.image_paths, self.label_paths = self.get_paths()
        
        self.images_common_transforms = tio.Compose([
                tio.Resample(self.img_space), 
                tio.Resize(self.img_size, image_interpolation= 'bspline'),
                tio.ZNormalization()
            ])
        
        self.labels_common_transforms = tio.Compose([
                tio.Resample(self.img_space),  
                tio.Resize(self.img_size, image_interpolation= 'nearest'),
            ])
        
        
    def get_paths(self):
        imagesTr = 'imagesTr'
        labelsTr = 'labelsTr'
        image_paths = [os.path.join(self.folder_path, imagesTr, image) for image in os.listdir(os.path.join(self.folder_path, imagesTr))]
        label_paths = [os.path.join(self.folder_path, labelsTr, image) for image in os.listdir(os.path.join(self.folder_path, labelsTr))]
    
        image_paths.sort(key=lambda x: int(x.split('_')[1]))
        label_paths.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
        if self.indices is not None:
            image_paths = [image_paths[i] for i in self.indices]
            label_paths = [label_paths[i] for i in self.indices]
    
        return image_paths, label_paths
    
    def __len__(self):
        return len(self.image_paths)
            
    def apply_common_transforms(self, image, label):
        transformed_image = self.images_common_transforms(image)
        transformed_label = self.labels_common_transforms(label)
        return transformed_image, transformed_label

    def __getitem__(self, idx):
        image_path, label_path = self.image_paths[idx], self.label_paths[idx]
        image = tio.ScalarImage(image_path)
        label = tio.LabelMap(label_path)
        transformed_image, transformed_label = self.apply_common_transforms(
            image, label)
        subject = tio.Subject(image = transformed_image, label = transformed_label)
        return subject
    
    def create_tio_datasets(self, train_samples=60, val_samples=2, test_samples=2, seed=42):
        total_length = len(self)
        
        assert train_samples + val_samples + test_samples <= total_length, "The sum of samples must not exceed the total length of the dataset"
        
        if seed is not None:
            random.seed(seed)
        
        indices = list(range(total_length))
        train_indices = random.sample(indices, train_samples)
        remaining_indices = set(indices) - set(train_indices)
        val_indices = random.sample(list(remaining_indices), val_samples)
        test_indices = list(remaining_indices - set(val_indices))[:test_samples]
    
        train_tio_dataset = tio.SubjectsDataset([self[idx] for idx in train_indices])
        val_tio_dataset = tio.SubjectsDataset([self[idx] for idx in val_indices])
        test_tio_dataset = tio.SubjectsDataset([self[idx] for idx in test_indices])
    
        return train_tio_dataset, val_tio_dataset, test_tio_dataset

    def create_grid_samplers(self, dataset, patch_size=None, full_img_size=True):
        
        if patch_size is None:
            if full_img_size:
                patch_size = self.img_size
        else:
            patch_size = (self.mean_image_size[0], self.mean_image_size[1]//3 + 1, self.mean_image_size[2])
            
        samplers = [tio.GridSampler(subject, patch_size) for subject in dataset]
        return samplers
    
    def augmentation_transforms(self):
        # num_of_aug_trasforms = num_of_aug_trasforms  
        all_transforms = [
            T.RandomFlip(),
            T.RandomAffine(),
            T.RandomElasticDeformation(),
            T.RandomAnisotropy(axes=(0, 1, 2), downsampling=(2, 5)),
            T.RandomMotion(),
            T.RandomGhosting(),
            T.RandomSpike(),
            T.RandomBiasField(),
            T.RandomBlur(),
            T.RandomNoise(),
            T.RandomSwap(),
            # T.RandomLabelsToImage(),
            T.RandomGamma()
        ]
        
        transform = tio.OneOf(all_transforms)
    
        # Shuffle the list of transforms
        # random.shuffle(all_transforms)
    
        # Select a random subset of transforms
        # selected_transforms = all_transforms[:num_of_aug_trasforms]
    
        # Compose the selected transforms
        # composed_transform = T.Compose(selected_transforms)
    
        return transform

    def aggregate_total_patches(self, samplers, is_train=False):
        patches_list = []
        for sampler in samplers:
            for patch in sampler:
                patches_list.append(patch)
                
        if is_train:
            transform = self.augmentation_transforms()
        else:
            transform = None
        
        patches_dataset = tio.SubjectsDataset(patches_list, transform=transform)
        # On the fly transforms are applied this way
        return patches_dataset
    
    # The structure of the dataset is the according one:
    # Sample 7: Subject(Keys: ('image', 'label', 'location'); images: 2)
    
#TODO: CHANGE BASED ON THE NEW FUNCTION
    def aggregated_train_val_plus_test_grid_sampler(self, train_samples=38, val_samples=2, test_samples=2, seed=42):
        
        train_tio_dataset, val_tio_dataset, test_tio_dataset = self.create_tio_datasets(train_samples= train_samples, val_samples= val_samples, test_samples= test_samples, seed= seed)
        
        train_grid_samplers = self.create_grid_samplers(train_tio_dataset)
        del train_tio_dataset
        val_grid_samplers = self.create_grid_samplers(val_tio_dataset)
        del val_tio_dataset
        test_grid_samplers = self.create_grid_samplers(test_tio_dataset)
        del test_tio_dataset
        
        patched_train_dataset = self.aggregate_total_patches(train_grid_samplers, is_train=True)
        del train_grid_samplers
        patched_val_dataset = self.aggregate_total_patches(val_grid_samplers)
        del val_grid_samplers
        
        return patched_train_dataset, patched_val_dataset, test_grid_samplers

#TODO: Could experiment with collate_fn
    def create_dataloaders(self, batch_size,  num_workers, train_samples=38, val_samples=2, test_samples=2, seed=42, shuffle=True):
        
        patched_train_dataset, patched_val_dataset, test_grid_samplers = \
            self.aggregated_train_val_plus_test_grid_sampler(train_samples=train_samples, val_samples=val_samples, test_samples=test_samples)
        
        train_dataloader = DataLoader(patched_train_dataset, batch_size= batch_size, shuffle=shuffle,
                                      num_workers=num_workers, pin_memory=True,
                                      drop_last=True)#, collate_fn=self.collate_fn)
            
        val_dataloader = DataLoader(patched_val_dataset, batch_size= batch_size, shuffle=shuffle,
                                      num_workers=num_workers, pin_memory=True,
                                      drop_last=True)#, collate_fn=self.collate_fn)
        
        return train_dataloader, val_dataloader, test_grid_samplers

# =============================================================================
# class PatchEmbedding(nn.Module):
#     def __init__(self, patch_size, in_channels, embed_dim):
#         super(PatchEmbedding, self).__init__()
#         self.patch_size = patch_size
#         self.in_channels = in_channels
#         self.emb_dim = embed_dim
#         # Convolution projectional layer
#         self.projection = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
#         
#     def forward(self, x):
#         # 3D convolutional to project input layers
#         projected_patches = self.projection(x)
#         # Flatten the spatial dimensions while retaining the embedding dimension
#         N, emb_dim, D_out, H_out, W_out = projected_patches.size()
#         # Reshape the output patches to flatten the spatial dimensions
#         flattened_patches = projected_patches.view(N, emb_dim, -1)
#         return flattened_patches
#         
# =============================================================================
            
# =============================================================================
# my_dataset = PatchedNiftiDatasetLoader('/home/georgeb/Desktop/diplomatiki/Dataset/')
# 
# 
# train_dtset, val_dtset, test_dtset = my_dataset.create_tio_datasets()
# 
# train_grid_samplers = my_dataset.create_grid_samplers(train_dtset)
# print(len(train_grid_samplers[1]))
# val_grid_samplers = my_dataset.create_grid_samplers(val_dtset)
# test_grid_samplers = my_dataset.create_grid_samplers(test_dtset)
# =============================================================================




        
        



# =============================================================================
# 
# train_patched_dtset = my_dataset.aggregate_total_patches(train_grid_samplers)
# val_patched_dtset = my_dataset.aggregate_total_patches(val_grid_samplers)
# =============================================================================


# output_dir = '/home/georgeb/Desktop/diplomatiki/Dataset/'

# train_dtset_basename = 'train_dtset.pt'
# val_dtset_basename = 'val_dtset.pt'

# train_dir = os.path.join(output_dir, train_dtset_basename)
# val_dir = os.path.join(output_dir, val_dtset_basename)

# my_dataset.save_patched_dataset(train_patched_dtset, train_dir)
# my_dataset.save_patched_dataset(val_patched_dtset, val_dir)
