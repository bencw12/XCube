import sys
import numpy as np
import fvdb
import torch
import polyscope as ps

chunk = np.load("all_chunks.npy")
points = torch.from_numpy(np.argwhere(chunk > 0)).int()
points = fvdb.JaggedTensor([points])
grid = fvdb.sparse_grid_from_ijk(points, voxel_sizes=0.08)

def visualize_grid(grid: fvdb.GridBatch):
    ps.set_allow_headless_backends(True)
    ps.init()
    xyz = grid.grid_to_world(grid.ijk.float())
    for batch_idx in range(grid.grid_count):
        pts = xyz[batch_idx].jdata.cpu().numpy()
        ps.register_point_cloud(f"grid_{batch_idx}", pts, radius=0.0025)
    ps.screenshot()

visualize_grid(grid)




# if __name__ == '__main__':
#     data_dir = sys.argv[1]
#     chunks = os.listdir(data_dir)
#     generate_train(chunks)
