import torch
import numpy as np
# from models.layers.gprojection_xyz import GProjection
import torch.nn.functional as F

batch = 1
c_dim = 1

feature_volume = torch.tensor(np.asarray(range(1, 46)).astype(np.float32).reshape(batch, c_dim, 5, 3, 3))#np.asarray(range(1, 46)).astype(np.float32).reshape(5, 3, 3)
#
# d, y, x = torch.meshgrid([torch.arange(0, 5, dtype=torch.float32),
#                             torch.arange(0, 3, dtype=torch.float32),
#                                torch.arange(0, 3, dtype=torch.float32)])
# grid = torch.stack([d, y, x], dim=-1).unsqueeze(0)
print("feature volume", feature_volume)
print("feature volume", feature_volume.size())
print("")
# print("d:", d, "\n", d.shape)
# print("")
# print("y:", y, "\n", y.shape)
# print("")
# print("x:", x, "\n", x.shape)

d, y, x = torch.tensor(1.), torch.tensor(0.), torch.tensor(-1.)
d = d.reshape(batch, 1, 1, 1)
y = y.reshape(batch, 1, 1, 1)
x = x.reshape(batch, 1, 1, 1)

grid = torch.stack([x, y, d], dim=-1)
print(grid.size())
print("grid.shape", grid.shape)
result = F.grid_sample(feature_volume, grid)
print("result:", result)
print("expected:", feature_volume[0, :, 4, 1, 0])

