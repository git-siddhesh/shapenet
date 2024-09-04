from tqdm import tqdm
import os

import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader  # Correct import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import random_split
from torch.cuda.amp import GradScaler, autocast

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", filename="sharpnet.log")
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

from utils import plot_3d_shape, save_point_cloud_ply, ResamplePoints, save_model, chamfer_loss, mmd_cd_loss, chamfer_loss_eval
from models import ResnetGenerator, NLayerDiscriminator

# Path to the dataset
DATASET_PATH = "./shapenetcore_partanno_segmentation_benchmark_v0_normal"
SPLIT_RATIO = 0.8
BATCH_SIZE = 1
EPOCHS = 1 #50

# Optimizers
BETA1 = 0.5
BETA2 = 0.999
LEARNING_RATE = 0.0002

# Initialize networks
INPUT_NC = 3
OUTPUT_NC = 3
NGF = 64
NDF = 64
NUM_BLOCKS = 6

# Training parameters
ACCUMULATION_STEPS = 1
batch_size = 1  # Start with a very small batch size
TARGET_SIZE = 64  # Reduced target size for smaller GPU usage

MODEL_PATH = "./saved_models"
os.makedirs(MODEL_PATH, exist_ok=True)
PLOT_PATH = "./plots"
os.makedirs(PLOT_PATH, exist_ok=True)



dataset = ShapeNet(root=DATASET_PATH, categories=["Airplane"]).shuffle()[:100]
# Provide the correct path to the extracted dataset
# dataset = ShapeNet(root=dataset_path, categories=["Table", "Lamp", "Guitar", "Motorbike"]).shuffle()[:1000]

logger.info(f"Number of Samples: {len(dataset)}")
logger.info(f"Sample: {dataset[0]}")

sample = dataset[0]
logger.info(f"Number of points: {sample.pos.shape[0]}, Dimension of each point: {sample.pos.shape[1]}")

#%%
# Visualize a sample
sample_idx = 9
# plot_3d_shape(dataset[sample_idx]) 
# NOTE: each data points have different shapes
save_point_cloud_ply(dataset[sample_idx], os.path.join(PLOT_PATH, f"point_cloud_{sample_idx}.ply"))
logger.info(f"Number of points: {dataset[sample_idx].pos.shape[0]}, Dimension of each point: {dataset[sample_idx].pos.shape[1]}")
#%% 
# Data Augmentation
augmentation = T.Compose([
    ResamplePoints(2048),
    T.RandomJitter(0.03),
    T.RandomFlip(axis=1),
    T.RandomShear(0.2)
])

# Apply augmentation
dataset.transform = augmentation

#%%
# DataLoader

# Split the dataset into train and test sets (80% train, 20% test)
train_size = int(SPLIT_RATIO * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders for train and test sets
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Verify the batch sizes
for batch in train_loader:
    print("Train Batch: ", batch)

for batch in test_loader:
    print("Test Batch: ", batch)

logger.info(f"Number of Training Samples: {len(train_dataset)}")
logger.info(f"Number of Test Samples: {len(test_dataset)}")
logger.info(f"Train dataset[0]: {train_dataset[0]}")
logger.info(f"Test dataset[0]: {test_dataset[0]}")


# Load a sample and apply augmentation
sample = next(iter(train_loader))
# plot_3d_shape(sample[0])
save_point_cloud_ply(sample[0], os.path.join(PLOT_PATH, "trainset_sample_point_cloud.ply")) 
transformed_sample = augmentation(sample[0])
# plot_3d_shape(transformed_sample)
save_point_cloud_ply(transformed_sample, os.path.join(PLOT_PATH, "trainset_sample_point_cloud_transformed.ply"))

# Print shapes to verify the transformations
logger.info(f"Original sample size: {sample[0].pos.size()}")
logger.info(f"Transformed sample size: {transformed_sample.pos.size()}")

#%%

# Initialize networks
input_nc = 3
output_nc = 3
ngf = 64
ndf = 64
num_blocks = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G_A2B = ResnetGenerator(INPUT_NC, OUTPUT_NC, NGF, NUM_BLOCKS).to(device)
G_B2A = ResnetGenerator(INPUT_NC, OUTPUT_NC, NGF, NUM_BLOCKS).to(device)
D_A = NLayerDiscriminator(INPUT_NC, NDF).to(device)
D_B = NLayerDiscriminator(INPUT_NC, NDF).to(device)


optimizer_G = Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=LEARNING_RATE, betas=(BETA1, BETA2))
optimizer_D_A = Adam(D_A.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
optimizer_D_B = Adam(D_B.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))


# Loss functions
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

batch_size = 1  # Start with a very small batch size

G_A2B.train()
G_B2A.train()
D_A.train()
D_B.train()
scaler = GradScaler()

for epoch in range(EPOCHS):
    total_loss_G = 0.0
    total_loss_D_A = 0.0
    total_loss_D_B = 0.0
    total_chamfer_distance_A = 0.0
    total_chamfer_distance_B = 0.0
    total_mmd_cd = 0.0

    for i, data in enumerate(tqdm(train_loader)):
        # Assuming 'data' is a PyTorch Geometric Data object
        real_A = data.pos.to(device).float()  # Convert to float
        real_B = data.batch.to(device).float()  # Convert to float

        # Calculate batch size and number of points
        batch_size = int(real_B.max().item() + 1)
        num_points = real_A.size(0) // batch_size

        # Ensure the calculated num_points is correct
        if real_A.size(0) % batch_size != 0:
            raise ValueError("The number of points is not divisible by batch size")

        real_A = real_A.view(batch_size, 3, num_points).contiguous()
        real_A = real_A.unsqueeze(3).permute(0, 1, 3, 2)  # Shape: [batch_size, 3, 1, num_points]
        real_B = real_B.view(batch_size, 1, num_points).expand(-1, 3, -1).unsqueeze(2)  # Shape: [batch_size, 3, 1, num_points]
        # Ensure the lengths match in the expected dimension
        if real_A.size(3) != real_B.size(3):
            min_size = min(real_A.size(3), real_B.size(3))
            real_A = real_A[:, :, :, :min_size]
            real_B = real_B[:, :, :, :min_size]

        # Resize inputs to smaller size suitable for your model
        real_A = F.interpolate(real_A, size=(TARGET_SIZE, TARGET_SIZE), mode='nearest')
        real_B = F.interpolate(real_B, size=(TARGET_SIZE, TARGET_SIZE), mode='nearest')

        # Zero gradients for all optimizers
        optimizer_G.zero_grad()
        optimizer_D_A.zero_grad()
        optimizer_D_B.zero_grad()

        # Generators A2B and B2A
        with autocast():
            real_A = real_A.half()  # Convert inputs to float16
            real_B = real_B.half()  # Ensure real_B is also float16

            fake_B = G_A2B(real_A)
            rec_A = G_B2A(fake_B)
            fake_A = G_B2A(real_B)
            rec_B = G_A2B(fake_A)

            # Identity loss
            idt_A = G_B2A(real_A)
            idt_B = G_A2B(real_B)
            loss_idt_A = criterion_identity(idt_A, real_A) * 5.0
            loss_idt_B = criterion_identity(idt_B, real_B) * 5.0

            # GAN loss
            pred_fake_B = D_B(fake_B)
            loss_G_A2B = criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))

            pred_fake_A = D_A(fake_A)
            loss_G_B2A = criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))

            # Cycle loss
            loss_cycle_A = criterion_cycle(rec_A, real_A) * 10.0
            loss_cycle_B = criterion_cycle(rec_B, real_B) * 10.0

            # Chamfer distance
            chamfer_distance_A = chamfer_loss(real_A, fake_A) * 0.001
            chamfer_distance_B = chamfer_loss(real_B, fake_B) * 0.001

            # MMD-CD loss
            mmd_cd = mmd_cd_loss(real_A, fake_A) * 0.001

            # Combined loss
            loss_G = (
                loss_G_A2B + loss_G_B2A + loss_cycle_A + loss_cycle_B +
                loss_idt_A + loss_idt_B + chamfer_distance_A + chamfer_distance_B + mmd_cd
            )
            loss_G = loss_G / ACCUMULATION_STEPS

        scaler.scale(loss_G).backward()
        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer_G)
            scaler.update()

        # Discriminator A
        with autocast():
            pred_real_A = D_A(real_A)
            loss_D_real_A = criterion_GAN(pred_real_A, torch.ones_like(pred_real_A))

            pred_fake_A = D_A(fake_A.detach())
            loss_D_fake_A = criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))

            loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5 / ACCUMULATION_STEPS
        scaler.scale(loss_D_A).backward()
        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer_D_A)
            scaler.update()

        # Discriminator B
        with autocast():
            pred_real_B = D_B(real_B)
            loss_D_real_B = criterion_GAN(pred_real_B, torch.ones_like(pred_real_B))

            pred_fake_B = D_B(fake_B.detach())
            loss_D_fake_B = criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))

            loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5 / ACCUMULATION_STEPS
        scaler.scale(loss_D_B).backward()
        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer_D_B)
            scaler.update()

        # Accumulate total losses for logging
        total_loss_G += loss_G.item()
        total_loss_D_A += loss_D_A.item()
        total_loss_D_B += loss_D_B.item()
        total_chamfer_distance_A += chamfer_distance_A.item()
        total_chamfer_distance_B += chamfer_distance_B.item()
        total_mmd_cd += mmd_cd.item()

    # Average losses and distances per epoch
    avg_loss_G = total_loss_G / len(train_loader)
    avg_loss_D_A = total_loss_D_A / len(train_loader)
    avg_loss_D_B = total_loss_D_B / len(train_loader)
    avg_chamfer_distance_A = total_chamfer_distance_A / len(train_loader)
    avg_chamfer_distance_B = total_chamfer_distance_B / len(train_loader)
    avg_mmd_cd = total_mmd_cd / len(train_loader)

    logger.info(f"Epoch [{epoch}/{EPOCHS}], Loss G: {avg_loss_G:.4f}, Loss D_A: {avg_loss_D_A:.4f}, Loss D_B: {avg_loss_D_B:.4f}")
    logger.info(f"Chamfer Distance A: {avg_chamfer_distance_A:.4f}, Chamfer Distance B: {avg_chamfer_distance_B:.4f}, MMD-CD: {avg_mmd_cd:.4f}")

    # Save models with epoch number directory
    if epoch % 10 == 0:
        save_dir = os.path.join(MODEL_PATH,  f"epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        save_model(G_A2B, os.path.join(save_dir, 'G_A2B.pth'))
        save_model(G_B2A, os.path.join(save_dir, 'G_B2A.pth'))
        save_model(D_A, os.path.join(save_dir, 'D_A.pth'))
        save_model(D_B, os.path.join(save_dir, 'D_B.pth'))



G_A2B.eval()
G_B2A.eval()
D_A.eval()
D_B.eval()
total_loss = 0
total_chamfer_A = 0
total_chamfer_B = 0
normalization_factor = 1e-3

for i, data in enumerate(tqdm(test_loader)):
    # Assuming 'data' is a PyTorch Geometric Data object
    real_A = data.pos.to(device).float()  # Convert to float
    real_B = data.batch.to(device).float()  # Convert to float

    # Calculate batch size and number of points
    batch_size = int(real_B.max().item() + 1)
    num_points = real_A.size(0) // batch_size

    # Ensure the calculated num_points is correct
    if real_A.size(0) % batch_size != 0:
        raise ValueError("The number of points is not divisible by batch size")

    real_A = real_A.view(batch_size, 3, num_points).contiguous()
    real_A = real_A.unsqueeze(3).permute(0, 1, 3, 2)  # Shape: [batch_size, 3, 1, num_points]
    real_B = real_B.view(batch_size, 1, num_points).expand(-1, 3, -1).unsqueeze(2)  # Shape: [batch_size, 3, 1, num_points]
    # Ensure the lengths match in the expected dimension
    if real_A.size(3) != real_B.size(3):
        min_size = min(real_A.size(3), real_B.size(3))
        real_A = real_A[:, :, :, :min_size]
        real_B = real_B[:, :, :, :min_size]

    # Resize inputs to smaller size suitable for your model
    real_A = F.interpolate(real_A, size=(TARGET_SIZE, TARGET_SIZE), mode='nearest')
    real_B = F.interpolate(real_B, size=(TARGET_SIZE, TARGET_SIZE), mode='nearest')

    # Zero gradients for all optimizers
    optimizer_G.zero_grad()
    optimizer_D_A.zero_grad()
    optimizer_D_B.zero_grad()

    with autocast():
        real_A = real_A.half()  # Convert inputs to float16
        real_B = real_B.half()

        fake_B = G_A2B(real_A)
        rec_A = G_B2A(fake_B)
        fake_A = G_B2A(real_B)
        rec_B = G_A2B(fake_A)

        # GAN loss
        pred_fake_B = D_B(fake_B)
        loss_G_A2B = criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))

        pred_fake_A = D_A(fake_A)
        loss_G_B2A = criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))

        # Cycle loss
        loss_cycle_A = criterion_cycle(rec_A, real_A) * 10.0
        loss_cycle_B = criterion_cycle(rec_B, real_B) * 10.0

        # Chamfer distance
        chamfer_distance_A = chamfer_loss_eval(real_A, fake_A) * normalization_factor
        chamfer_distance_B = chamfer_loss_eval(real_B, fake_B) * normalization_factor

        # MMD-CD loss
        mmd_cd = mmd_cd_loss(real_A, fake_A) * normalization_factor

        # Combined loss
        loss_G = (loss_G_A2B + loss_G_B2A + loss_cycle_A + loss_cycle_B + chamfer_distance_A + chamfer_distance_B + mmd_cd)
        total_loss += loss_G.item()

        # Accumulate Chamfer distances
        total_chamfer_A += chamfer_distance_A.item()
        total_chamfer_B += chamfer_distance_B.item()

        # Clear GPU memory
        del fake_B, fake_A, rec_A, rec_B, loss_G
        torch.cuda.empty_cache()

total_loss /= len(test_loader)
avg_chamfer_A = total_chamfer_A / len(test_loader)
avg_chamfer_B = total_chamfer_B / len(test_loader)

logger.info(f"Test Loss: {total_loss:.4f}")
logger.info(f"Chamfer Distance A: {avg_chamfer_A:.4f}")
logger.info(f"Chamfer Distance B: {avg_chamfer_B:.4f}")
logger.info(f"MMD_CD: {mmd_cd:.4f}")

# save the final model
save_dir = os.path.join(MODEL_PATH, "final")
os.makedirs(save_dir, exist_ok=True)
save_model(G_A2B, os.path.join(save_dir, 'G_A2B_final.pth'))
save_model(G_B2A, os.path.join(save_dir, 'G_B2A_final.pth'))
save_model(D_A, os.path.join(save_dir, 'D_A_final.pth'))
save_model(D_B, os.path.join(save_dir, 'D_B_final.pth'))
