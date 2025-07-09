import os
import numpy as np
import SimpleITK as sitk
from utils.preprocessing import *


patient_id = 'Patient_1'
num = int(patient_id.split('_')[1])

dwi_path = f'D:\\Thesis\\dataset_registered\\MOBI_DWI\\b900\\{patient_id}\\result.1.nii.gz'
adc_path = f'D:\\Thesis\\dataset_registered\\MOBI_ADC\\{patient_id}\\{patient_id}.nii.gz'
t1_path = f'D:\\Thesis\\dataset_registered\\MOBI_T1W_TSE\\{patient_id}.nii.gz'
lbl_path = f'D:\\Thesis\\whole_spine_mask\\BOT_0{num:02d}.nii.gz'
res_path = f'C:\\Repos\\Thesis\\RESULTS'


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

# Relabel the connected components in the spine mask
print("Relabeling connected components")
# Find connected components in the label image
connected_components = sitk.ConnectedComponent(lbl)
# Ensure valid and sorted labels based on Y-axis 
lbl_img, unique_labels = relabel_components(connected_components, os.path.join(res_path, 'spine_mask.nii.gz'), min_voxels=1000, std_threshold=2.0)

# Crop the images to the bounding box of the labels
dwi_spine_img = sitk.Mask(dwi, lbl_img)
adc_spine_img = sitk.Mask(adc, lbl_img)
t1_spine_img = sitk.Mask(t1, lbl_img)
sitk.WriteImage(dwi_spine_img, os.path.join(res_path, 'dwi_spine_img.nii.gz'))
sitk.WriteImage(adc_spine_img, os.path.join(res_path, 'adc_spine_img.nii.gz'))
sitk.WriteImage(t1_spine_img, os.path.join(res_path, 't1_spine_img.nii.gz'))

# Process the ADC image
print("\nLocating suspicious values from ADC image")
# Keep only the voxels within the range [0.6, 1.2] and segment them
adc_array = sitk.GetArrayFromImage(adc_spine_img)
segmented_adc_array = np.where((adc_array >= 0.6) & (adc_array <= 1.2), 1, 0)
segmented_adc_image = sitk.GetImageFromArray(segmented_adc_array.astype(np.uint8))
segmented_adc_image.CopyInformation(adc)
# Find the different connected components in the segmented ADC mask
adc_connected_components, unique_adc_labels = get_valid_lesions(segmented_adc_image, os.path.join(res_path, 'adc_lesion.nii.gz'))


# Process the DWI image
print("\nLocating suspicious values from DWI image")
# Apply Otsu thresholding to the DWI image
otsu = sitk.OtsuMultipleThresholdsImageFilter()
otsu.SetNumberOfThresholds(4)
otsu.SetNumberOfHistogramBins(256)
otsu_mask = otsu.Execute(dwi_spine_img)
# Keep only the highest two thresholds (3 and 4)
otsu_les3 = otsu_mask == 3
otsu_les4 = otsu_mask == 4
otsu_les = otsu_les3 + otsu_les4
# Find the different connected components in the segmented DWI mask
dwi_connected_components, unique_dwi_labels = get_valid_lesions(otsu_les, os.path.join(res_path, 'dwi_lesion.nii.gz'))


# Process the T1 image
print("\nLocating suspicious values from T1 image")
otsu = sitk.OtsuMultipleThresholdsImageFilter()
otsu.SetNumberOfThresholds(4)
otsu.SetNumberOfHistogramBins(256)
otsu_mask = otsu.Execute(t1_spine_img)
# Keep only the lowest threshold (1)
otsu_les = otsu_mask == 1
# Find the different connected components in the segmented T1 mask
t1_connected_components, unique_t1_labels = get_valid_lesions(otsu_les, os.path.join(res_path, 't1_lesion.nii.gz'))


# Combine the masks by intersecting the voxels that are present in at least two of the three images
print("\nCombining the masks")
t1_bin = (t1_connected_components > 0).astype(np.uint8)
dwi_bin = (dwi_connected_components > 0).astype(np.uint8)
adc_bin = (adc_connected_components > 0).astype(np.uint8)
comb_mask = ((t1_bin + dwi_bin + adc_bin) >= 2).astype(np.uint8)
comb_img = sitk.GetImageFromArray(comb_mask)
comb_img.CopyInformation(t1_spine_img)
# Find the different connected components in the combined mask
comb_mask_components, unique_comb_labels = get_valid_lesions(comb_img, os.path.join(res_path, 'combined_mask.nii.gz'))


# Apply morphological operations to the combined mask
print("\nApplying morphological operations")
img = sitk.GetImageFromArray(comb_mask_components)
img.CopyInformation(t1_spine_img)
# Apply binary morphological closing to fill small holes in the mask
filtered = sitk.BinaryMorphologicalClosing(img, [1, 1, 1])  # radius in voxels
# Apply Gaussian smoothing to the binary image
gaussian = sitk.SmoothingRecursiveGaussian(sitk.Cast(filtered > 0, sitk.sitkFloat32), sigma=1.0)
gaussian_binary = sitk.Cast(gaussian > 0.3, sitk.sitkUInt8)
# Crop the image to the shape of the labels
gaussian_binary = sitk.Mask(gaussian_binary, lbl_img)
# Find the different connected components in the final mask
final_connected_components, unique_final_labels = get_valid_lesions(gaussian_binary, os.path.join(res_path, 'final_mask.nii.gz'))