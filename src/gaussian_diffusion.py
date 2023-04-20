import torch
import torch.nn as nn
import torch.nn.functional as F



def extract(coef, time, n_dims):
    coef_t = coef[time]
    while coef_t.dim() < n_dims:
        coef_t = coef_t.unsqueeze(-1)
    return coef_t


from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDiffusion(nn.Module):
    def __init__(self, beta, posterior_var_type='beta'):
        assert posterior_var_type in ['beta', 'beta_tilde']
        super().__init__()
        self.beta = beta
        self.n_timesteps = beta.shape[0]

        alpha = 1 - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0)
        alpha_cumprod_prev = torch.cat([torch.ones(1), alpha_cumprod[:-1]], dim=0)


        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(alpha_cumprod))
        self.register_buffer('sqrt_one_minus_alpha_cumprod', torch.sqrt(1 - alpha_cumprod))


        self.register_buffer('sqrt_alpha', torch.sqrt(alpha))
        self.register_buffer('eps_coef', beta / - torch.sqrt(1 - alpha_cumprod))

        
        if posterior_var_type == 'beta':
            self.register_buffer('posterior_var', beta)
        elif posterior_var_type == 'beta_tilde':
            self.register_buffer('posterior_var', beta * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod))


    def q_sample(self, x, time, noise=None):
        if noise is None:
            noise = torch.randn_like(x)

        x_coef = extract(self.sqrt_alpha_cumprod, time, x.dim())
        noise_coef = extract(self.sqrt_one_minus_alpha_cumprod, time, x.dim())

        return x_coef * x + noise_coef * noise
    

    def training_loss(self, model, x):
        time = torch.randint(0, self.n_timesteps, (x.shape[0],), device=x.device)
        noise = torch.randn_like(x)
        x_t = self.q_sample(x, time, noise)
        pred = model(x_t, time)
        loss = F.mse_loss(pred, noise)

        return loss
    

    def posterior_mean(self, model, x, time):
        eps = model(x, time)
        eps_coef = extract(self.eps_coef, time, eps.dim())
        sqrt_alpha_t = extract(self.sqrt_alpha, time, eps.dim())
        mu = (x + eps_coef * eps) / sqrt_alpha_t

        return mu
    
    @torch.no_grad()
    def p_sample_step(self, model, x, time):
        mu = self.posterior_mean(model, x, time)
        if time[0].item() == 0:
            return mu
        else:
            noise = torch.randn_like(mu)
            var = extract(self.posterior_var, time, mu.dim())
            return mu + torch.sqrt(var) * noise
        
    @torch.no_grad()
    def sample(self, model, shape):
        device = self.posterior_var.device
        x = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(self.n_timesteps))):
            time = torch.ones(shape[0], dtype=torch.long, device=device) * t
            x = self.p_sample_step(model, x, time)

        return x