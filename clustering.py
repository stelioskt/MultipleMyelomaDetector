import os
import numpy as np
import SimpleITK as sitk
from utils.preprocessing import *


patient_id = 'Patient_20'
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



lbl_np = sitk.GetArrayFromImage(lbl_img)
unique_vertebrae = np.unique(lbl_np)
unique_vertebrae = unique_vertebrae[unique_vertebrae != 0]  # Exclude background
print(f"Unique vertebrae labels: {unique_vertebrae}")








from skimage.segmentation import slic
from skimage.util import img_as_float
dwi_arr   = sitk.GetArrayFromImage(dwi_spine_img)
lbl_arr   = sitk.GetArrayFromImage(lbl_img) > 0

# Ensure float range
vol_float = img_as_float(dwi_arr)

n_segments    = 1000       # adjust for desired granularity
compactness   = 0.1        # balance spatial vs. intensity

# Perform 3D SLIC supervoxel segmentation
print(f"Running SLIC: n_segments={n_segments}, compactness={compactness}...")
supervoxels = slic(
    vol_float,
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
print(f"Saved supervoxels to {out_path}")























# from skimage.segmentation import slic
# from skimage.color import gray2rgb
# from skimage.measure import regionprops
# from sklearn.cluster import SpectralClustering
# import matplotlib.pyplot as plt

# dwi_np = sitk.GetArrayFromImage(dwi_spine_img)
# vertebra_masks = np.zeros_like(dwi_np)

# for v_id in unique_vertebrae:
#     print(f"\nProcessing vertebra ID: {v_id}")
#     vertebra_mask = (lbl_np == v_id).astype(np.uint8)
#     vertebra_dwi = dwi_np * vertebra_mask  # masked DWI

#     # Optional: crop around the vertebra for faster processing
#     z_indices = np.any(vertebra_mask, axis=(1,2))
#     z_min, z_max = np.where(z_indices)[0][[0, -1]]

#     for z in range(z_min, z_max + 1):
#         if np.sum(vertebra_mask[z]) < 50: continue  # skip empty slices

#         slice_img = vertebra_dwi[z]
#         slice_mask = vertebra_mask[z]
#         rgb_slice = gray2rgb(slice_img)

#         segments = slic(rgb_slice, n_segments=200, compactness=10, sigma=1, start_label=1)
#         props = regionprops(segments, intensity_image=slice_img)

#         if len(props) < 4: continue  # too few regions

#         features = np.array([[r.mean_intensity, *r.centroid] for r in props])
#         clustering = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', assign_labels='kmeans')
#         cluster_labels = clustering.fit_predict(features)

#         # Assign clustered labels back to superpixels
#         clustered_slice = np.zeros_like(segments)
#         for i, r in enumerate(props):
#             clustered_slice[segments == r.label] = cluster_labels[i]

#         # Choose one cluster to mark as tumor (e.g., highest mean intensity)
#         mean_ints = [np.mean(slice_img[clustered_slice == cl]) for cl in np.unique(cluster_labels)]
#         tumor_cluster = np.argmax(mean_ints)
#         tumor_mask_slice = (clustered_slice == tumor_cluster).astype(np.uint8)

#         vertebra_masks[z] += tumor_mask_slice  # accumulate

#         # Optional: visualize
#         if z % 5 == 0:
#             plt.figure(figsize=(10,3))
#             plt.subplot(1,3,1); plt.imshow(slice_img, cmap='gray'); plt.title(f'Vertebra {v_id} - Slice {z}')
#             plt.subplot(1,3,2); plt.imshow(segments, cmap='nipy_spectral'); plt.title('SLIC')
#             plt.subplot(1,3,3); plt.imshow(tumor_mask_slice, cmap='hot'); plt.title('Tumor Mask')
#             plt.show()















# # Process the ADC image
# print("\nLocating suspicious values from ADC image")
# # Keep only the voxels within the range [0.6, 1.2] and segment them
# adc_array = sitk.GetArrayFromImage(adc_spine_img)
# segmented_adc_array = np.where((adc_array >= 0.6) & (adc_array <= 1.2), 1, 0)
# segmented_adc_image = sitk.GetImageFromArray(segmented_adc_array.astype(np.uint8))
# segmented_adc_image.CopyInformation(adc)
# # Find the different connected components in the segmented ADC mask
# adc_connected_components, unique_adc_labels = get_valid_lesions(segmented_adc_image, os.path.join(res_path, 'adc_lesion.nii.gz'))


# # Process the DWI image
# print("\nLocating suspicious values from DWI image")
# # Apply Otsu thresholding to the DWI image
# otsu = sitk.OtsuMultipleThresholdsImageFilter()
# otsu.SetNumberOfThresholds(4)
# otsu.SetNumberOfHistogramBins(256)
# otsu_mask = otsu.Execute(dwi_spine_img)
# # Keep only the highest two thresholds (3 and 4)
# otsu_les3 = otsu_mask == 3
# otsu_les4 = otsu_mask == 4
# otsu_les = otsu_les3 + otsu_les4
# # Find the different connected components in the segmented DWI mask
# dwi_connected_components, unique_dwi_labels = get_valid_lesions(otsu_les, os.path.join(res_path, 'dwi_lesion.nii.gz'))


# # Process the T1 image
# print("\nLocating suspicious values from T1 image")
# otsu = sitk.OtsuMultipleThresholdsImageFilter()
# otsu.SetNumberOfThresholds(4)
# otsu.SetNumberOfHistogramBins(256)
# otsu_mask = otsu.Execute(t1_spine_img)
# # Keep only the lowest threshold (1)
# otsu_les = otsu_mask == 1
# # Find the different connected components in the segmented T1 mask
# t1_connected_components, unique_t1_labels = get_valid_lesions(otsu_les, os.path.join(res_path, 't1_lesion.nii.gz'))


# # Combine the masks by intersecting the voxels that are present in at least two of the three images
# print("\nCombining the masks")
# t1_bin = (t1_connected_components > 0).astype(np.uint8)
# dwi_bin = (dwi_connected_components > 0).astype(np.uint8)
# adc_bin = (adc_connected_components > 0).astype(np.uint8)
# comb_mask = ((t1_bin + dwi_bin + adc_bin) >= 2).astype(np.uint8)
# comb_img = sitk.GetImageFromArray(comb_mask)
# comb_img.CopyInformation(t1_spine_img)
# # Find the different connected components in the combined mask
# comb_mask_components, unique_comb_labels = get_valid_lesions(comb_img, os.path.join(res_path, 'combined_mask.nii.gz'))


# # Apply morphological operations to the combined mask
# print("\nApplying morphological operations")
# img = sitk.GetImageFromArray(comb_mask_components)
# img.CopyInformation(t1_spine_img)
# # Apply binary morphological closing to fill small holes in the mask
# filtered = sitk.BinaryMorphologicalClosing(img, [1, 1, 1])  # radius in voxels
# # Apply Gaussian smoothing to the binary image
# gaussian = sitk.SmoothingRecursiveGaussian(sitk.Cast(filtered > 0, sitk.sitkFloat32), sigma=1.0)
# gaussian_binary = sitk.Cast(gaussian > 0.3, sitk.sitkUInt8)
# # Crop the image to the shape of the labels
# gaussian_binary = sitk.Mask(gaussian_binary, lbl_img)
# # Find the different connected components in the final mask
# final_connected_components, unique_final_labels = get_valid_lesions(gaussian_binary, os.path.join(res_path, 'final_mask.nii.gz'))









# # ip_path = '/media/georgeb/4 TB WD/PhD/Data/harmonized_data_registration/MOBI_T1_DIXON/IP/Patient_8.nii.gz'
# # op_path = '/media/georgeb/4 TB WD/PhD/Data/harmonized_dataset_transformations/MOBI_T1_DIXON/OP/Patient_8/result.nii'

# # # ff_img = compute_fat_fraction(ip_path, op_path, os.path.join(res_path, 'fat_fraction_without_abs.nii.gz'))
# # # ff_img = sitk.WriteImage(ff_img, os.path.join(res_path, 'fat_fraction.nii.gz'))

# # with_abs_ff_img_path = os.path.join(res_path, 'fat_fraction_with_abs.nii.gz')
# # # without_abs_ff_img_path = os.path.join(res_path, 'fat_fraction_without_abs.nii.gz')

# # ff_img = sitk.ReadImage(with_abs_ff_img_path)
# # ff_img, _ = resample_image(ff_img, target_spacing)
# # print(ff_img.GetSize())
# # print(segmented_adc_image.GetSize())
# # ff_img_masked = sitk.Mask(ff_img, segmented_adc_image)
# # ff_array = sitk.GetArrayFromImage(ff_img_masked)

# # segmented_ff_array = np.where((ff_array <= 0.1), 0, 1)
# # segmented_ff_mask = sitk.GetImageFromArray(segmented_ff_array.astype(np.uint8))
# # # Align the segmented fat fraction mask with the original image
# # segmented_ff_mask.CopyInformation(ff_img)