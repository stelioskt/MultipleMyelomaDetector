import os
import numpy as np
import SimpleITK as sitk

from Botis.preprocessing import resample_image
# from ff_gen import *

# DWI initial segmentation
img_path = 'D:/Thesis/dataset_registered/MOBI_DWI/b900/Patient_20/Patient_20.nii.gz'
lbl_path = 'D:/Thesis/whole_spine_mask/BOT_020.nii.gz'
res_path = 'C:/Repos/Thesis/RESULTS'

# This function is used on the T1 DIXON images, using ip and op channels to compute the fat fraction
def calculate_fat_fraction_sitk(ip_image, op_image):
    """
    Calculate fat fraction images using Dixon IP and OP images with SimpleITK.

    Parameters:
        ip_image (sitk.Image): In-phase image.
        op_image (sitk.Image): Out-of-phase image.

    Returns:
        sitk.Image: Fat fraction image.
    """
    # Convert SimpleITK images to NumPy arrays
    ip_array = sitk.GetArrayFromImage(ip_image).astype(np.float32)
    op_array = sitk.GetArrayFromImage(op_image).astype(np.float32)

    # Calculate water and fat signals
    water_signal = (ip_array + op_array) / 2
    fat_signal = (ip_array - op_array) / 2

    # Calculate fat fraction
    fat_fraction = fat_signal / (water_signal + fat_signal + 1e-8)  # Avoid division by zero
    fat_fraction = np.clip(fat_fraction, 0, 1)  # Clip values to [0, 1]

    # Convert back to SimpleITK image
    fat_fraction_image = sitk.GetImageFromArray(fat_fraction)
    fat_fraction_image.CopyInformation(ip_image)  # Preserve spatial metadata

    return fat_fraction_image

if not os.path.exists(res_path):
    os.makedirs(res_path)
# We use an isotropic spacing to our images as a good practice for the Adaptive Histogram Equalization and Otsu algorithm application
# Define the new size and isotropic spacing
target_spacing = [1.0, 1.0, 1.0]

img = sitk.ReadImage(img_path)
lbl = sitk.ReadImage(lbl_path)

# Preprocess the image
resampled_img,_ = resample_image(img, target_spacing)

# For the labels we use k nearest in order to preserve the labels without distorions
resampled_lbl,_ = resample_image(lbl, target_spacing, 'nearest')
resampled_img = sitk.Mask(resampled_img, resampled_lbl)

# This function divides all the regions that preserve connectivity between them
connected_components = sitk.ConnectedComponent(resampled_lbl)
sitk.WriteImage(connected_components, os.path.join(res_path, 'connected_components.nii.gz'))

# This step may be redundant
relabeled_components = sitk.RelabelComponent(connected_components)
sitk.WriteImage(relabeled_components, os.path.join(res_path, 'relabeled_components.nii.gz'))

# Extracting the number of vertebrae and the sacrum (iero osto, einai katw apo toys spondulous)
lbl_ar = sitk.GetArrayFromImage(relabeled_components)
unique_labels = np.unique(lbl_ar)
print("Unique labels in lbl_image:", unique_labels)

# Orizoume ton otsu algorithmo me 4 thresholds
otsu = sitk.OtsuMultipleThresholdsImageFilter()
otsu.SetNumberOfThresholds(4)

# Dimiourgoume to template pou tha kanoume paste tis alloioseis, pairnei ws attributes ta stoixeia tou resampled_img
total_mask = sitk.Image(resampled_img.GetSize(), sitk.sitkUInt8)
total_mask.CopyInformation(resampled_img)
for i in range(1, len(unique_labels)):
    # gia kathe spondulo
    vrt = lbl_ar == i
    print(f"Processing label {i}")
    vrt_lbl = sitk.GetImageFromArray(vrt.astype(np.uint8))
    vrt_lbl.CopyInformation(resampled_lbl)

    # Get bounding box, (orismos enos bounding box gurw apo kathe spondulo)
    print("Getting bounding box")
    vrt_stats = sitk.LabelShapeStatisticsImageFilter()
    vrt_stats.Execute(vrt_lbl)
    bb = vrt_stats.GetBoundingBox(1)

    # Extract the ROI (eksagogi mono auths ths perioxis gia thn efarmogi tou filtrou)
    print("Extracting ROI")
    roi = sitk.RegionOfInterest(resampled_img, bb[3:], bb[:3])
    # Z normalization
    roi = sitk.Normalize(roi)
    # Crop the binary mask to match the ROI
    cropped_vrt_lbl = sitk.RegionOfInterest(vrt_lbl, bb[3:], bb[:3])
    # Ensure cropped_vrt_lbl has the same physical space as roi
    cropped_vrt_lbl.CopyInformation(roi)

    # Apply the mask, (wste to filtro na efarmostei mono ston spondulo kai oxi stis peroixes tou bounding box pou den periexoun plhroforia)
    roi_mask = sitk.Mask(roi, cropped_vrt_lbl)
    # Adaptive histogram equalization
    print("Applying adaptive histogram equalization")
    # anakatanemei wraia to istogramma auth h methodos kai voitha sto segmentation twn upoptwn perioxwn
    roi_mask = sitk.AdaptiveHistogramEqualization(roi_mask)
    # Otsu thresholding
    print("Applying Otsu thresholding")
    otsu_mask = otsu.Execute(roi_mask)
    otsu_les = otsu_mask == 4
    # Append Otsu result to total mask
    print("Appending Otsu result to total mask")
    total_mask = sitk.Paste(total_mask, otsu_les, otsu_les.GetSize(), [0, 0, 0], bb[:3])
    # sitk.WriteImage(roi, os.path.join(res_path, f'vrt_{i}_get_region.nii.gz'))

sitk.WriteImage(total_mask, os.path.join(res_path, 'total_mask.nii.gz'))

# adc_img_path = '/media/georgeb/4 TB WD/PhD/Data/Registered_Imgs/MOBI_ADC/Patient_8/Patient_8.nii.gz'
# adc_img = sitk.ReadImage(adc_img_path)
# r_adc, _ = resample_image(adc_img, target_spacing)

# # Apply the total mask to the ADC image
# masked_adc = sitk.Mask(r_adc, total_mask)

# # Convert the masked ADC image to a NumPy array
# adc_array = sitk.GetArrayFromImage(masked_adc)

# # Keep only the voxels within the range [0.6, 1.2] and segment them
# segmented_adc_array = np.where((adc_array >= 0.6) & (adc_array <= 1.2), 1, 0)

# # Convert the segmented array back to a SimpleITK image
# segmented_adc_image = sitk.GetImageFromArray(segmented_adc_array.astype(np.uint8))
# segmented_adc_image.CopyInformation(r_adc)  # Preserve spatial metadata

# # Save the segmented ADC image
# sitk.WriteImage(segmented_adc_image, os.path.join(res_path, 'segmented_adc.nii.gz'))

# ip_path = '/media/georgeb/4 TB WD/PhD/Data/harmonized_data_registration/MOBI_T1_DIXON/IP/Patient_8.nii.gz'
# op_path = '/media/georgeb/4 TB WD/PhD/Data/harmonized_dataset_transformations/MOBI_T1_DIXON/OP/Patient_8/result.nii'

# # ff_img = compute_fat_fraction(ip_path, op_path, os.path.join(res_path, 'fat_fraction_without_abs.nii.gz'))
# # ff_img = sitk.WriteImage(ff_img, os.path.join(res_path, 'fat_fraction.nii.gz'))

# with_abs_ff_img_path = os.path.join(res_path, 'fat_fraction_with_abs.nii.gz')
# # without_abs_ff_img_path = os.path.join(res_path, 'fat_fraction_without_abs.nii.gz')

# ff_img = sitk.ReadImage(with_abs_ff_img_path)
# ff_img, _ = resample_image(ff_img, target_spacing)
# print(ff_img.GetSize())
# print(segmented_adc_image.GetSize())
# ff_img_masked = sitk.Mask(ff_img, segmented_adc_image)
# ff_array = sitk.GetArrayFromImage(ff_img_masked)

# segmented_ff_array = np.where((ff_array <= 0.1), 0, 1)
# segmented_ff_mask = sitk.GetImageFromArray(segmented_ff_array.astype(np.uint8))
# # Align the segmented fat fraction mask with the original image
# segmented_ff_mask.CopyInformation(ff_img)

# min_r = 2.5
# min_vol = np.pi * (min_r ** 3)

# cc = sitk.ConnectedComponent(segmented_ff_mask)
# print(type(cc))  # Debugging: Check if cc is a SimpleITK image

# rlbl_fil = sitk.RelabelComponentImageFilter()
# rlbl_fil.SetMinimumObjectSize(int(min_vol))
# fil_lbl = rlbl_fil.Execute(cc)
# print(type(fil_lbl))  # Debugging: Check if fil_lbl is a SimpleITK image

# # Ensure the final mask has the same spatial metadata as the original image
# fil_lbl.CopyInformation(ff_img)

# sitk.WriteImage(fil_lbl, os.path.join(res_path, 'with_abs_final_mask.nii.gz'))