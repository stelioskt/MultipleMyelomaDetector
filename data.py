import os
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
import SimpleITK as sitk
from torch_geometric.data import Data
from radiomics import featureextractor, logger
import logging
import time

from clustering import clustering


logger.setLevel(logging.ERROR)
logging.getLogger('radiomics').setLevel(logging.ERROR)

# Configure PyRadiomics extractor
def configure_radiomics(bin_width=25, spacing=None, enabled_features=None):
    settings = {
        'binWidth': bin_width,
        'resampledPixelSpacing': spacing,
        'enableCExtensions': True,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    # Enable core feature classes
    classes = enabled_features or [
        'firstorder', 'shape', 'glcm', 'glrlm', 'glszm'     # 'gldm', 'ngtdm'
    ]
    for cls in classes:
        extractor.enableFeatureClassByName(cls)
    return extractor

# Instantiate a global extractor
radiomics_extractor = configure_radiomics()


def extract_radiomics(vol, mask, extractor):
    """
    Compute radiomics features for a single volume/mask pair.
    """
    # Convert to SimpleITK images
    sitk_vol = sitk.GetImageFromArray(vol.astype(np.float32))
    sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    # Extract features
    result = extractor.execute(sitk_vol, sitk_mask)
    # Filter out metadata keys and sort by feature name for consistent ordering
    feats = {k: v for k, v in result.items() if k.startswith('original_')}
    # Sort and return values
    keys = sorted(feats.keys())
    return np.array([feats[k] for k in keys], dtype=np.float32)


def create_graph(patient_id):
    start = time.time()
    supervoxels, t1_arr, t2_arr, sv_labels, sv_comps, sv_centroids = clustering(patient_id)
    print(f"Clustering took {time.time() - start:.2f} seconds")
    N = int(supervoxels.max())

    node_feats = []
    node_labels = []

    for lab in tqdm(range(1, N+1), desc=f"{patient_id}"):
        mask = (supervoxels == lab)
        if not mask.any():
            continue

        # Binary lesion label: 1 if any voxel in this supervoxel is lesion
        is_lesion = int(bool(sv_labels[lab-1]))
        node_labels.append(is_lesion)

        # Extract radiomics features for each modality
        mask_uint = mask.astype(np.uint8)
        feats_t1 = extract_radiomics(t1_arr, mask_uint, radiomics_extractor)
        feats_t2 = extract_radiomics(t2_arr, mask_uint, radiomics_extractor)

        # Concatenate T1 and T2 feature vectors
        feat = np.hstack([feats_t1, feats_t2])  # shape (F1 + F2)
        node_feats.append(feat)

    # Stack into torch tensors
    x = torch.tensor(np.stack(node_feats, axis=0), dtype=torch.float)
    y = torch.tensor(np.array(node_labels, dtype=np.int64), dtype=torch.long)
    comp = torch.tensor(sv_comps, dtype=torch.long)
    centroids = torch.tensor(sv_centroids, dtype=torch.float)

    # Build 6-connectivity adjacency
    adj = defaultdict(set)
    for axis in (0, 1, 2):
        rolled = np.roll(supervoxels, -1, axis=axis)
        boundary = (supervoxels != rolled)
        a_ids = supervoxels[boundary]
        b_ids = rolled[boundary]
        for u, v in zip(a_ids.flat, b_ids.flat):
            if u > 0 and v > 0 and u != v:
                adj[u-1].add(v-1)
                adj[v-1].add(u-1)

    # Flatten into edge_index
    edges = [[u, v] for u, nbrs in adj.items() for v in nbrs]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Create the PyG Data object
    data = Data(x=x, edge_index=edge_index, y=y, comp=comp, pos=centroids)

    print(f"Graph created with {data.num_nodes} nodes and {data.num_edges} edges")
    print(f"Lesion percentage: {data.y.sum().item() / data.num_nodes:.2%}")

    return data


if __name__ == '__main__':
    lbl_dir = "Dataset/Labels"
    out_dir = "Dataset/Graphs"
    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(lbl_dir):
        if fname.endswith('.nii.gz'):
            patient_id = fname.split('.')[0]
            print(f"\n=================================")
            print(f"Processing patient {patient_id}...")
            try:
                graph = create_graph(patient_id)
                torch.save(graph, os.path.join(out_dir, f"{patient_id}.pt"))
            except Exception as e:
                print(f"Error processing {patient_id}: {e}")
