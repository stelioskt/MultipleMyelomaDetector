import numpy as np
import SimpleITK as sitk

def resample_image(image, target_spacing, interpolation='linear'):
    """
    Resample a SimpleITK image to a new spacing.
    """
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


def calculate_fat_fraction_sitk(ip_image, op_image):
    """
    Calculate fat fraction images using T1 Dixon IP and OP images with SimpleITK.

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


def relabel_components(image, save_path=None, std_threshold=2.0, min_voxels = 1000):
    """
    Relabels connected components based on Y-axis position (head to pelvis),
    calculates volume and centroid, and optionally saves the relabeled image.
    
    Args:
        image (sitk.Image): Input label image (components labeled).
        save_path (str, optional): Path to save the relabeled image.
        sd_threshold (float, optional): Standard deviation threshold for filtering components.
        min_voxels (int, optional): Minimum number of voxels for a component to be considered valid.
    Returns:
        sitk.Image: New relabeled image.
    """
    # Get the unique labels in the image
    lbl_array = sitk.GetArrayFromImage(image)
    unique_labels = np.unique(lbl_array)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background (label 0)

    # Find centroids of each component
    centroids = []
    for label in unique_labels:
        mask = lbl_array == label
        coords = np.argwhere(mask)  # coords: [Z, Y, X]
        mean_z = np.mean(coords[:, 0])
        mean_y = np.mean(coords[:, 1])
        mean_x = np.mean(coords[:, 2])
        num_voxels = len(coords)
        centroids.append((label, [mean_z, mean_y, mean_x], num_voxels))
    
    # Calculate mean and standard deviation of volumes
    volumes = np.array([num_voxels for _, _, num_voxels in centroids])
    mean_volume = np.mean(volumes)
    std_volume = np.std(volumes)
    
    low_volume = mean_volume - std_threshold * std_volume
    high_volume = mean_volume + std_threshold * std_volume
    print(f"Mean volume: {mean_volume}, Std volume: {std_volume}")
    print(f"Volume threshold: {low_volume} to {high_volume}")
    
    # Filter components based on volume threshold
    valid_centroids = []
    for label, mean, num_voxels in centroids:
        if (num_voxels >= min_voxels) and (low_volume <= num_voxels <= high_volume):
            valid_centroids.append((label, mean, num_voxels))
    
    # Sort components by Y coordinate (head to feet)
    centroids_sorted = sorted(valid_centroids, key=lambda x: x[1][1])

    # Prepare new label array
    reordered_lbl_array = np.zeros_like(lbl_array)
    
    # Calculate voxel volume
    spacing = image.GetSpacing()
    voxel_volume = spacing[0] * spacing[1] * spacing[2]  # mm³ per voxel

    print("\nComponent details (after Y-axis relabeling):")
    print("Label | Volume (mm³) | Centroid (x, y, z) ")
    
    for new_label, (old_label, mean, num_voxels) in enumerate(centroids_sorted, start=1):
        component_mask = lbl_array == old_label

        # Define volume and centroid
        volume = num_voxels * voxel_volume
        mean_z, mean_y, mean_x = mean[0], mean[1], mean[2]

        print(f"{new_label:5d} | {volume:8.2f} mm³ | ({mean_x:.2f}, {mean_y:.2f}, {mean_z:.2f})")

        # Assign new label
        reordered_lbl_array[component_mask] = new_label

    # Convert back to SimpleITK image
    unique_labels = np.unique(reordered_lbl_array)
    reordered_image = sitk.GetImageFromArray(reordered_lbl_array)
    reordered_image.CopyInformation(image)
    reordered_image = sitk.Cast(reordered_image, sitk.sitkUInt8)


    # Save the new relabeled image if a path is given
    if save_path:
        sitk.WriteImage(reordered_image, save_path)

    return reordered_image, unique_labels


def get_valid_lesions(image, save_path=None, min_diameter_mm=5.0, min_volume=200):
    """
    Filters and relabels connected components from an ADC-based mask based on physical size and volume.
    Components are sorted and relabeled by increasing volume.

    Args:
        image (sitk.Image): Input connected component label image.
        min_diameter_mm (float): Minimum physical diameter in mm.
        min_volume (int): Minimum voxel volume.

    Returns:
        tuple: (valid_labels_array, unique_labels)
    """
    connected_components = sitk.ConnectedComponent(image)
    
    spacing = connected_components.GetSpacing()  # (x, y, z)
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(connected_components)

    img_array = sitk.GetArrayFromImage(connected_components)
    valid_components = []

    for label in label_stats.GetLabels():
        if label == 0:
            continue  # Skip background label

        bbox = label_stats.GetBoundingBox(label)  # (x, y, z, size_x, size_y, size_z)
        size_mm = bbox[3:]
        volume = label_stats.GetPhysicalSize(label)

        if volume > min_volume and all(s >= min_diameter_mm for s in size_mm):
            valid_components.append((label, size_mm, volume))

    # Sort valid components by volume
    valid_components.sort(key=lambda x: x[2])  # sort by volume

    # Relabel sorted components
    valid_labels = np.zeros_like(img_array)
    print("Label |  Size (mm) : dx, dy, dz  | Volume (voxels)")
    for new_label, (label, size_mm, volume) in enumerate(valid_components, start=1):
        valid_labels[img_array == label] = new_label
        print(f"{new_label:5d} | ({size_mm[0]:5.2f}, {size_mm[1]:5.2f}, {size_mm[2]:5.2f}) mm | {volume:5.2f}")

    # Convert back to SimpleITK image
    unique_labels = np.unique(valid_labels)
    relabelled_image = sitk.GetImageFromArray(valid_labels.astype(np.uint8))
    relabelled_image.CopyInformation(connected_components)
    relabelled_image = sitk.Cast(relabelled_image, sitk.sitkUInt8)

    if save_path:
        sitk.WriteImage(relabelled_image, save_path)

    return valid_labels, unique_labels