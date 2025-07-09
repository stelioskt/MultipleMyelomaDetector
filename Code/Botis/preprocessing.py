import os
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize

def resample_image(image, target_spacing, interpolation='linear'):
    original_spacing = np.array(image.GetSpacing(), dtype=np.float32)
    original_size = np.array(image.GetSize(), dtype=int)
    
    # Calculate the new size based on the target spacing
    new_size = np.round(original_size * (original_spacing / target_spacing)).astype(int)
    
    # Resample the image using SimpleITK
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size.tolist())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkLinear if interpolation == 'linear' else sitk.sitkNearestNeighbor)
    
    resampled_image = resample.Execute(image)
    
    return resampled_image, new_size

def resize_image(image, target_size, interpolation='linear'):
    # Convert the image to a numpy array
    image_array = sitk.GetArrayFromImage(image)
    
    # Resize the image using skimage
    resized_array = resize(image_array, target_size, mode='edge', anti_aliasing=False, order=1 if interpolation == 'linear' else 0)
    
    # Convert the resized array back to a SimpleITK image
    resized_image = sitk.GetImageFromArray(resized_array)
    resized_image.SetSpacing(image.GetSpacing())
    resized_image.SetDirection(image.GetDirection())
    resized_image.SetOrigin(image.GetOrigin())
    
    return resized_image

def preprocess_image(image_path, target_spacing, target_size, interpolation='linear'):
    """
    Preprocess an image by resampling to target spacing, resizing to target size, and normalizing intensity values via z-normalization.
    """
    image = sitk.ReadImage(image_path)
    
    # Resample the image to the target spacing
    resampled_image, new_size = resample_image(image, target_spacing, interpolation)
    
    # Resize the image to the target size
    resized_image = resize_image(resampled_image, target_size, interpolation)
    
    # Normalize the intensity values
    normalized_image = sitk.Normalize(resized_image)
    
    return normalized_image, new_size

def preprocess_label(label_path, target_spacing, target_size, new_size):
    """
    Preprocess a label by resampling to target spacing and resizing to target size.
    """
    label = sitk.ReadImage(label_path)
    
    # Resample the label to the target spacing using nearest neighbor interpolation
    resampled_label, _ = resample_image(label, target_spacing, interpolation='nearest')
    
    # Resize the label to the target size using nearest neighbor interpolation
    resized_label = resize_image(resampled_label, target_size, interpolation='nearest')
    
    return resized_label

base_path = 'D:/Thesis'
img_path = f'{base_path}/dataset_registered/MOBI_DWI/b900/Patient_20/result.1.nii/result.1.nii'
img_path_2 = f'{base_path}/dataset_registered/MOBI_ADC/Patient_20/Patient_20.nii/Patient_20.nii'
lbl_path = f"{base_path}/whole_spine_mask/BOT_020.nii/BOT_020.nii"
res_path = "C:/Repos/Thesis/RESULTS"

# Define the new size and isotropic spacing
target_size = [64, 512, 256]
target_spacing = [1.0, 1.0, 1.0]

img = sitk.ReadImage(img_path)
img_2 = sitk.ReadImage(img_path_2)
lbl = sitk.ReadImage(lbl_path)

# Preprocess the image
img,_ = resample_image(img, target_spacing)
img_2,_ = resample_image(img_2, target_spacing)
lbl,_ = resample_image(lbl, target_spacing, 'nearest')

img = sitk.Mask(img, lbl)
img_2 = sitk.Mask(img_2, lbl)
sitk.WriteImage(img, os.path.join(res_path, 'img.nii.gz'))
sitk.WriteImage(img_2, os.path.join(res_path, 'img_2.nii.gz'))
sitk.WriteImage(lbl, os.path.join(res_path, 'lbl.nii.gz'))

cc = sitk.ConnectedComponent(lbl)
sitk.WriteImage(cc, os.path.join(res_path, 'connected_components.nii.gz'))

rc = sitk.RelabelComponent(cc)
sitk.WriteImage(rc, os.path.join(res_path, 'relabeled_components.nii.gz'))

print('rc type:', type(rc))

lbl_ar = sitk.GetArrayFromImage(rc)
vrt_nums = np.unique(lbl_ar)

for vrt_num in vrt_nums:
    vrt = lbl_ar == vrt_num
    # Convert vrt to a SimpleITK image
    vrt_img = sitk.GetImageFromArray(vrt.astype(np.uint8))
    vrt_img.CopyInformation(lbl)
    # Get bounding box
    vrt_stats = sitk.LabelShapeStatisticsImageFilter()
    vrt_stats.Execute(vrt_img)
    bb = vrt_stats.GetBoundingBox(1)
    # Write label
    vrt = sitk.GetImageFromArray(vrt.astype(np.uint8))
    vrt = sitk.WriteImage(vrt, os.path.join(res_path, f'vrt_{vrt_num}_lbl.nii.gz'))
    # Roi extraction
    roi = sitk.RegionOfInterest(img, bb[3:], bb[:3])
    roi_adc = sitk.RegionOfInterest(img_2, bb[3:], bb[:3])
    roi = sitk.Normalize(roi)
    roi_adc = sitk.Normalize(roi_adc)
    sitk.WriteImage(roi, os.path.join(res_path, f'vrt_{vrt_num}_get_region.nii.gz'))
    sitk.WriteImage(roi_adc, os.path.join(res_path, f'vrt_{vrt_num}_get_region_adc.nii.gz'))