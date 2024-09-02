import torch
import plotly.express as px
from plyfile import PlyData, PlyElement
import numpy as np

# Visualization function
def plot_3d_shape(shape):
    print("Number of data points: ", shape.pos.shape[0])
    fig = px.scatter_3d(x=shape.pos[:, 0], y=shape.pos[:, 1], z=shape.pos[:, 2],
                       color=shape.y)
    fig.show()
    x = shape.pos[:, 0].cpu().numpy()
    y = shape.pos[:, 1].cpu().numpy()
    z = shape.pos[:, 2].cpu().numpy()
    fig = px.scatter_3d(x=x, y=y, z=z, opacity=0.3)
    fig.show()

# Function to save point cloud as PLY
def save_point_cloud_ply(shape, filename="point_cloud.ply"):
    # Extract coordinates
    x = shape.pos[:, 0].cpu().numpy()
    y = shape.pos[:, 1].cpu().numpy()
    z = shape.pos[:, 2].cpu().numpy()
    labels = shape.y.cpu().numpy() if shape.y is not None else None

    # Prepare data for PLY file
    vertex = np.array([(x[i], y[i], z[i]) for i in range(len(x))],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    
    if labels is not None:
        vertex = np.array([(x[i], y[i], z[i], labels[i]) for i in range(len(x))],
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('label', 'u1')])
    
    # Create PlyElement
    ply_element = PlyElement.describe(vertex, 'vertex')
    
    # Write to PLY file
    PlyData([ply_element], text=True).write(filename)
    print(f"Point cloud saved to {filename}")

# Example usage
# save_point_cloud_ply(shape, "point_cloud.ply")


class ResamplePoints:
    def __init__(self, num_points):
        self.num_points = num_points

    def __call__(self, data):
        pos = data.pos
        if pos.size(0) > self.num_points:
            # If more points, randomly sample `self.num_points` points
            idx = torch.randperm(pos.size(0))[:self.num_points]
        elif pos.size(0) < self.num_points:
            # If fewer points, randomly duplicate some points to make `self.num_points`
            idx = torch.randint(0, pos.size(0), (self.num_points,))
        else:
            return data
        data.pos = pos[idx]
        if data.x is not None:
            data.x = data.x[idx]
        if data.y is not None:
            data.y = data.y[idx]
        return data
    

def save_model(model, path):
    """Saves the model to the specified path."""
    torch.save(model.state_dict(), path)

def chamfer_loss(real, fake):
    # Define Chamfer loss (example implementation)
    diff = real - fake
    return torch.sum(diff ** 2)

def chamfer_loss_eval(real, fake):
    diff = real - fake
    return torch.sum(diff ** 2, dim=-1).mean()

def mmd_cd_loss(real, fake):
    # Calculate Chamfer distance
    chamfer_dist = chamfer_loss(real, fake)
    # Maximum Mean Discrepancy component
    real_mean = torch.mean(real, dim=1)
    fake_mean = torch.mean(fake, dim=1)
    mmd = torch.mean((real_mean - fake_mean) ** 2)
    return chamfer_dist + mmd
