import torch
from torch.autograd import grad
import torch.nn.functional as F

from models import ChamferDistance

# WGAN loss functions with gradient penalty
def dis_loss_fake(discriminator, real_image, fake_image, label):
    fake_predict = discriminator(fake_image).mean()

    batch_size = real_image.size(0)
    feature_size_real = real_image.size(1) * real_image.size(2) if len(real_image.size()) > 2 else real_image.size(1)
    feature_size_fake = fake_image.size(1) * fake_image.size(2) if len(fake_image.size()) > 2 else fake_image.size(1)

    real_image = real_image.view(batch_size, feature_size_real)
    fake_image = fake_image.view(batch_size, feature_size_fake)

    max_size = max(feature_size_real, feature_size_fake)
    if feature_size_real < max_size:
        real_image = F.pad(real_image, (0, max_size - feature_size_real))
    if feature_size_fake < max_size:
        fake_image = F.pad(fake_image, (0, max_size - feature_size_fake))

    eps = torch.rand(batch_size, 1).to(real_image.device)
    x_hat = eps * real_image + (1 - eps) * fake_image
    x_hat.requires_grad = True

    hat_predict = discriminator(x_hat)
    grad_x_hat = grad(outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
    grad_penalty = ((grad_x_hat.view(batch_size, -1).norm(2, dim=1) - 1) ** 2).mean()
    grad_penalty = 10 * grad_penalty

    fake_loss = fake_predict + grad_penalty

    return fake_loss

def dis_loss_real(discriminator, real_image, label):
    real_image = real_image.view(real_image.size(0), -1)
    real_predict = discriminator(real_image).mean() - 0.001 * (discriminator(real_image) ** 2).mean()
    return -real_predict

def gen_loss(discriminator, fake_image, label):
    fake_image = fake_image.view(fake_image.size(0), -1)
    loss = -discriminator(fake_image).mean()
    return loss

# Initialize Chamfer distance
chamfer_loss = ChamferDistance()

# Example usage in training loop
def combined_loss(generator, discriminator, encoder, real_data, labels, device):
    # Extract tensor data from real_data if it's a Data object
    if hasattr(real_data, 'x'):
        real_data = real_data.x

    # Generate fake data
    batch_size = real_data.size(0)
    gen_input = torch.randn(batch_size, 64).to(device)
    fake_data = generator(gen_input)

    # Calculate WGAN losses
    real_loss = dis_loss_real(discriminator, real_data)
    fake_loss = dis_loss_fake(discriminator, real_data, fake_data)
    g_loss = gen_loss(discriminator, fake_data)

    # Calculate Chamfer distance
    encoded_real = encoder(real_data)
    chamfer_distance = chamfer_loss(encoded_real, fake_data)

    # Combine losses
    total_d_loss = real_loss + fake_loss
    total_g_loss = g_loss + chamfer_distance

    return total_d_loss, total_g_loss