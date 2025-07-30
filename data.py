import numpy as np
import torch
from collections import defaultdict
import os
from torch_geometric.data import Data
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split

from clustering import clustering


def create_graph(patient_id):
    supervoxels, t1_arr, t2_arr, ls_arr = clustering(patient_id)

    N = int(supervoxels.max())

    node_feats = []
    node_labels = []

    def compute_quintiles(vol, mask):
        """Return the five 20%-increments over vol[mask]."""
        return np.percentile(vol[mask], [20,40,60,80,100])

    for lab in tqdm(range(1, N+1)):
        mask = (supervoxels == lab)
        if not mask.any():
            continue

        q1 = compute_quintiles(t1_arr, mask)
        q2 = compute_quintiles(t2_arr, mask)
        print(f"Supervoxel {lab}: T1 quintiles={q1}, T2 quintiles={q2}")
        feat = np.hstack([q1, q2])  # shape (10,)
        node_feats.append(feat)

        # Binary lesion label: 1 if any voxel in this supervoxel is lesion
        is_lesion = bool(ls_arr[lab-1])
        node_labels.append(int(is_lesion))

    # Stack into torch tensors
    node_feats_np = np.stack(node_feats, axis=0)
    x = torch.from_numpy(node_feats_np).float()     # [N, 10]
    #TODO: Select feature extraction method (pyradiomics)
    node_labels_np = np.array(node_labels, dtype=np.int64)
    y = torch.from_numpy(node_labels_np)    # [N]

    # Build 6-connectivity adjacency       #TODO: Select connectivity method
    adj = defaultdict(set)
    for axis in (0,1,2):
        rolled = np.roll(supervoxels, -1, axis=axis)
        boundary = (supervoxels != rolled)
        a_ids = supervoxels[boundary]
        b_ids = rolled[boundary]
    for u, v in zip(a_ids.flat, b_ids.flat):
        # only connect supervoxels inside your mask (labels 1…N)
        if u > 0 and v > 0 and u != v:
            adj[u-1].add(v-1)
            adj[v-1].add(u-1)

    # Flatten into edge_index
    edges = [[u, v] for u, nbrs in adj.items() for v in nbrs]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, E]

    # Create the PyG Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    
    #TODO: To be defined if one graph per sequence will be used or one unified graph

    print(f"Graph created with {data.num_nodes} nodes and {data.num_edges} edges")
    print(f"Lesion precentage: {data.y.sum().item()/data.num_nodes:.2%}")
    
    return data

lbl_dir = "Dataset/Labels"
out_dir = "Dataset/Graphs"
for fname in os.listdir(lbl_dir):
    full_path = os.path.join(lbl_dir, fname)
    if os.path.isfile(full_path):
        patient_id = fname.split('.')[0]
        print(f"Processing patient {patient_id}...")
        try:
            data = create_graph(patient_id)
            torch.save(data, os.path.join(out_dir, f"{patient_id}.pt"))
        except Exception as e:
            print(f"Error processing {patient_id}: {e}")