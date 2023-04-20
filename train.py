
from tqdm import tqdm

import torch
from torchvision.datasets import MNIST, CelebA
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader

from src.unet import Unet
from src.noise_schedule import linear_noise_schedule
from src.gaussian_diffusion import GaussianDiffusion
from torch_ema import ExponentialMovingAverage



# scale the images to [-1, 1]
transform = transforms.Compose([
    transforms.CenterCrop(140),
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)
])

#train_data = NIST(root='../../data', train=True, download=True, transform=transform)
train_data = CelebA(root='../../data', split='train', download=True, transform=transform)
train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=8, drop_last=True)

device = torch.device('cuda:0')

model = Unet(
    dim=32,
    dim_mults=(1, 2, 4, 8),
    channels=3,
).to(device)

beta = linear_noise_schedule(start=1e-4, end=2e-2, n_timesteps=1000)
gd = GaussianDiffusion(beta).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)


for epoch in range(100):
    print(f'Epoch {epoch}')
    epoch_loss = 0.

    for i, (x, _) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        x = x.to(device)
        loss = gd.training_loss(model, x)
        loss.backward()
        optimizer.step()
        ema.update(model.parameters())
        epoch_loss += loss.item()

    print(f'Loss: {epoch_loss / len(train_dataloader)}')


with ema.average_parameters():
    torch.save(model.state_dict(), 'ckpts/unet_celeba.pth')