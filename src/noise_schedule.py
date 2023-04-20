import torch


def linear_noise_schedule(start=1e-4, end=2e-2, n_timesteps=1000):
    return torch.linspace(start, end, n_timesteps)

