import os
import numpy as np
import SimpleITK as sitk
from skimage.segmentation import slic
from utils.preprocessing import *


patient_id = 'Patient_1'
num = int(patient_id.split('_')[1])

dwi_path = f"Dataset/Images/DWI_b900/{patient_id}.nii.gz"
adc_path = f"Dataset/Images/ADC/{patient_id}.nii.gz"
t1_path = f"Dataset/Images/T1W-TSE/{patient_id}.nii.gz"
lbl_path = f"Dataset/Masks/BOT_0{num:02d}.nii.gz"
res_path = "Results"


if not os.path.exists(res_path):
    os.makedirs(res_path)

# Read the images
dwi = sitk.ReadImage(dwi_path)
adc = sitk.ReadImage(adc_path)
t1 = sitk.ReadImage(t1_path)
lbl = sitk.ReadImage(lbl_path)

# Define the new size and isotropic spacing
target_spacing = [1.0, 1.0, 1.0]

# Resample the images
dwi,_ = resample_image(dwi, target_spacing)
adc,_ = resample_image(adc, target_spacing)
t1,_ = resample_image(t1, target_spacing)
sitk.WriteImage(dwi, os.path.join(res_path, 'dwi_img.nii.gz'))
sitk.WriteImage(adc, os.path.join(res_path, 'adc_img.nii.gz'))
sitk.WriteImage(t1, os.path.join(res_path, 't1_img.nii.gz'))
img_dims = dwi.GetSize()
print(f"Resampled dwi image size: {img_dims[0]} x {img_dims[1]} x {img_dims[2]}")

# Resample the labels
# We use k nearest in order to preserve the labels without distorions
lbl,_ = resample_image(lbl, target_spacing, 'nearest')
lbl.CopyInformation(dwi)
lbl = sitk.Cast(lbl, sitk.sitkUInt8)

# Relabel the connected components in the label image
print("Relabeling connected components")
connected_components = sitk.ConnectedComponent(lbl)
lbl_img, unique_labels = relabel_components(connected_components, os.path.join(res_path, 'spine_mask.nii.gz'), min_voxels=1000, std_threshold=2.0)

# Crop the images to the bounding box of the labels
dwi_spine_img = sitk.Mask(dwi, lbl_img)
adc_spine_img = sitk.Mask(adc, lbl_img)
t1_spine_img = sitk.Mask(t1, lbl_img)
sitk.WriteImage(dwi_spine_img, os.path.join(res_path, 'dwi_spine_img.nii.gz'))
sitk.WriteImage(adc_spine_img, os.path.join(res_path, 'adc_spine_img.nii.gz'))
sitk.WriteImage(t1_spine_img, os.path.join(res_path, 't1_spine_img.nii.gz'))


# ========== Supervoxel Segmentation =========

dwi_arr   = sitk.GetArrayFromImage(dwi_spine_img)
lbl_arr   = sitk.GetArrayFromImage(lbl_img) > 0

n_segments    = 1000       # adjust for desired granularity
compactness   = 0.1        # balance spatial vs. intensity

# Perform 3D SLIC supervoxel segmentation
print(f"Running SLIC: n_segments={n_segments}, compactness={compactness}...")
supervoxels = slic(
    dwi_arr,
    n_segments=n_segments,
    compactness=compactness,
    mask=lbl_arr,
    channel_axis=None,
    start_label=1
)
print(f"Generated {supervoxels.max()} supervoxels")

# Convert supervoxel map back to SimpleITK image
supervox_img = sitk.GetImageFromArray(supervoxels.astype(np.int32))
supervox_img.CopyInformation(dwi_spine_img)

# Save supervoxel labels
out_path = os.path.join(res_path, 'supervoxels.nii.gz')
sitk.WriteImage(supervox_img, out_path)
print("Supervoxels saved")