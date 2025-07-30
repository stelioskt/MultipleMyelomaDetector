import os
import numpy as np
import SimpleITK as sitk
from skimage.segmentation import slic

from utils.preprocessing import resample_image, relabel_components


def clustering(patient_id, min_lesion_vox=10, min_fraction=0.1, max_voxel_diff=2):
    num = int(patient_id.split('_')[1])

    # Paths
    t1_path     = f"Dataset/Images/T1W-TSE/{patient_id}.nii.gz"
    t2_path     = f"Dataset/Images/STIR/{patient_id}.nii.gz"
    lbl_path    = f"Dataset/Masks/BOT_{num:03d}.nii.gz"
    lesion_path = f"Dataset/Labels/{patient_id}.nii.gz"
    res_path    = "Results"

    os.makedirs(res_path, exist_ok=True)

    # Read images
    t1 = sitk.ReadImage(t1_path)
    t2 = sitk.ReadImage(t2_path)
    lbl = sitk.ReadImage(lbl_path)
    lesion = sitk.ReadImage(lesion_path)

    # Define isotropic spacing
    target_spacing = [1.0, 1.0, 1.0]

    # Resample images
    t1,_ = resample_image(t1, target_spacing) 
    t2,_ = resample_image(t2, target_spacing)
    lbl,_ = resample_image(lbl, target_spacing, 'nearest')
    lesion, _ = resample_image(lesion, target_spacing, 'nearest')



    # Ensure mask metadata aligns with T1
    lbl.CopyInformation(t1)
    lesion.CopyInformation(t1)
    lbl = sitk.Cast(lbl, sitk.sitkUInt8)

    # Relabel connected spine components
    # REVIEW: This step could be ommited for the moment
    connected_components = sitk.ConnectedComponent(lbl)
    lbl_img = relabel_components(
        connected_components,
        os.path.join(res_path, 'spine_mask.nii.gz'),
        min_voxels=1000, 
        std_threshold=2.0
    )

    # ================== Align T1 and T2 ==================
    sh1 = t1.GetSize()[::-1]
    sh2 = t2.GetSize()[::-1]
    # Compute per-axis difference
    diffs = [sh1[i] - sh2[i] for i in range(3)]
    abs_diffs = [abs(d) for d in diffs]
    print(f"[{patient_id}] Pre-mask shapes: T1 {sh1}, T2 {sh2}, diffs {diffs}")

    # Skip if any difference > max_voxel_diff
    if any(d > max_voxel_diff for d in abs_diffs):
        print(f"[{patient_id}] Size mismatch exceeds {max_voxel_diff} voxels: {abs_diffs} → skipping")
        return None, None, None, None

    # Determine crop size = min shape
    min_shape = tuple(min(sh1[i], sh2[i]) for i in range(3))
    # ROI filter expects size (X,Y,Z)
    roi = sitk.RegionOfInterestImageFilter()
    roi.SetIndex((0,0,0))
    roi.SetSize((min_shape[2], min_shape[1], min_shape[0]))

    # Crop volumes and masks
    t1 = roi.Execute(t1)
    t2 = roi.Execute(t2)
    lbl_img = roi.Execute(lbl_img)
    lesion  = roi.Execute(lesion)

    # After cropping, align metadata to T1 grid
    t2.CopyInformation(t1)
    lbl_img.CopyInformation(t1)
    lesion.CopyInformation(t1)

    # Now mask to spine on cropped volumes
    t1_spine_img = sitk.Mask(t1, lbl_img)
    t2_spine_img = sitk.Mask(t2, lbl_img)

    # ========= Supervoxel Segmentation =========
    t1_arr = sitk.GetArrayFromImage(t1_spine_img)
    t2_arr = sitk.GetArrayFromImage(t2_spine_img)
    lbl_arr = sitk.GetArrayFromImage(lbl_img) > 0

    #TODO: Maybe lower the segments - hyperparameter tuning

    n_segments = 1000
    compactness = 0.1
    print(f"Running SLIC: n_segments={n_segments}, compactness={compactness}...")
    supervoxels = slic(
        t1_arr,
        n_segments=n_segments,
        compactness=compactness,
        mask=lbl_arr,
        channel_axis=None,
        start_label=1
    )
    print(f"Generated {supervoxels.max()} supervoxels")

    # Save supervoxels
    supervox_img = sitk.GetImageFromArray(supervoxels.astype(np.int32))
    supervox_img.CopyInformation(t1_spine_img)
    sitk.WriteImage(supervox_img, os.path.join(res_path, f'supervoxels_{patient_id}.nii.gz'))

    # ========= Label Extraction =========
    lesion_arr = sitk.GetArrayFromImage(lesion) > 0
    num_nodes  = int(supervoxels.max())
    sv_labels  = np.zeros(num_nodes, dtype=np.uint8)

    for lab in range(1, num_nodes+1):
        mask = (supervoxels == lab)
        region_size = mask.sum()
        if region_size == 0:
            continue
        lesion_count = int(lesion_arr[mask].sum())
        if lesion_count >= min_lesion_vox and (lesion_count / region_size) >= min_fraction:
            sv_labels[lab-1] = 1
            
    #TODO: Optimize the merging of supervoxels that contain lesion voxels
    
    return supervoxels, t1_arr, t2_arr, sv_labels
