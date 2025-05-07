import sys
import numpy as np
import fvdb
import torch
import polyscope as ps
import argparse
import os
from tqdm import tqdm

args = argparse.ArgumentParser()
args.add_argument('--data_dir', type=str, default='./data/minecraft_raw', help='dir for raw data')
args.add_argument('--target_dir', type=str, default='./data/minecraft/', help='dir for processed data')
# args.add_argument('--num_vox', type=int, default=16, help='dimension of voxel grid')
args.add_argument('--pct_train', type=int, default=80, help='integer percentage of dataset to be used for training. test and validation sets are each (100 - pct_train)/2 percent of the data')
args = args.parse_args()

data_dir = args.data_dir
target_dir = args.target_dir

raw_chunks = os.listdir(data_dir)
n_chunks = len(raw_chunks)

pct_test = (100 - args.pct_train) // 2
pct_val = pct_test

raw_chunks = np.array(raw_chunks)
np.random.shuffle(raw_chunks)

n_test = int(n_chunks * (pct_test / 100.0))
n_val = int(n_chunks * (pct_val / 100.0))
n_train = n_chunks - (n_test + n_val)

train = raw_chunks[0:n_train]
test = raw_chunks[n_train:n_train + n_test]
val = raw_chunks[n_train+n_test: ]

os.makedirs(target_dir, exist_ok=True)
for split in ['train', 'test', 'val']:
    # list of ids for the dataloader
    with open(f"{target_dir}/{split}.lst", "w") as f:
        for label in eval(split):
            label = label.strip(".npy")
            f.write(f'{label}\n')

voxel_size = 0.01 # for some reason this is what works, i think it just depends on the value in the config
# actually process the data
for chunk in tqdm(raw_chunks):
    data = np.load(f"{data_dir}/{chunk}", allow_pickle=True)
    # print(data.max())
    origin = [voxel_size / 2.0] * 3
    points = torch.from_numpy(np.argwhere(data > 0)).int()
    points = fvdb.JaggedTensor([points])
    grid = fvdb.sparse_grid_from_ijk(points, voxel_sizes=voxel_size, origins=origin)
    xyz = grid.grid_to_world(grid.ijk.float()).jdata
    min_val = xyz.min()
    max_val = xyz.max()
    xyz = (xyz - min_val) / (max_val - min_val)
    target_grid = fvdb.sparse_grid_from_points(fvdb.JaggedTensor(xyz),
                                               voxel_sizes=voxel_size,
                                               origins=origin)

    save_dict = {
        "points": target_grid.to("cpu")
    }

    label = chunk.strip(".npy")
    torch.save(save_dict, f'{target_dir}/{label}.pkl')
