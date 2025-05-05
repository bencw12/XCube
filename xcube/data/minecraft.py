import os
from xcube.data.base import DatasetSpec as DS
from xcube.data.base import RandomSafeDataset
import torch
import numpy as np
import fvdb

class MinecraftDataset(RandomSafeDataset):
    def __init__(self, data_path, spec, split, resolution, custom_name="minecraft", random_seed=0, skip_on_error=False, hparams=None, duplicate_num=1):
        if isinstance(random_seed, str):
            super().__init__(0, True, skip_on_error)
        else:
            super().__init__(random_seed, False, skip_on_error)

        self.data_path = data_path
        self.resolution = resolution
        self.skip_on_error = skip_on_error
        self.custom_name = custom_name
        self.spec = spec
        self.split = split
        # assuming all chunks are in the base dir
        self.chunks = os.listdir(data_path)
        self.hparams = hparams
        self.duplicate_num = duplicate_num

    def __len__(self):
        return len(self.chunks) & self.duplicate_num

    def get_name(self):
        return f"{self.custom_name}-{self.split}"

    def get_short_name(self):
        return self.custom_name

    def _get_item(self, data_id, rng):
        chunk_id = data_id % len(self.chunks)
        chunk = self.chunks[chunk_id]
        data = {}
        input_data = np.load(f"{self.data_path}/{chunk}")
        x, y, z = input_data.shape

        vox_size = self.hparams['voxel_size'][0]

        print("GET ITEM\n", vox_size)
        origin = [vox_size / 2.0] * 3

        points = torch.from_numpy(np.argwhere(input_data > 0)).int()

        points = fvdb.JaggedTensor([points])

        grid = fvdb.sparse_grid_from_ijk(points, voxel_sizes=vox_size, origins=origin)
        xyz = grid.grid_to_world(grid.ijk.float()).jdata
        xyz_norm = xyz * 128/100
        target_grid = fvdb.sparse_grid_from_points(fvdb.JaggedTensor(xyz_norm),
                                                   voxel_sizes=vox_size,
                                                   origins = origin)
        # target_normal = input_points.splat_trilinear(fvdb.)

        if DS.SHAPE_NAME in self.spec:
            data[DS.SHAPE_NAME] = str(chunk_id)

        if DS.INPUT_PC in self.spec:
            data[DS.INPUT_PC] = target_grid

        if DS.GT_DENSE_PC in self.spec:
            data[DS.GT_DENSE_PC] = target_grid

        normal  = torch.zeros((384 * 16 * 16, 3))
        if DS.TARGET_NORMAL in self.spec:
            data[DS.TARGET_NORMAL] = normal

        if DS.GT_DENSE_NORMAL in self.spec:
            data[DS.GT_DENSE_NORMAL] = normal

        return data
