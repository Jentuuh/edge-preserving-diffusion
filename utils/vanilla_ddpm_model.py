import torch 
import numpy as np
import decimal 
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import save_png, save_video

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class ddpm_diffusion_trainer(nn.Module):
    def __init__(self, config, model, device):
        super().__init__()

        self.model = model
        self.num_steps = config.diffusion.num_steps
        self.register_buffer('betas', torch.linspace(config.diffusion.beta_min, config.diffusion.beta_max, self.num_steps).double().to(device))

        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # To compute q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar).to(device))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar).to(device))
        
        
    def sample_prior_x_T(self, train_dataset, batch_size, device, save_forward_imgs=False, sample_dir=None):

        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        x_0 = next(iter(dataloader))[0].to(device)
        save_png(sample_dir, x_0.cpu(), "x_0.png")

        if save_forward_imgs:
            full_path = torch.empty((self.num_steps + 1, x_0.shape[0], x_0.shape[1], x_0.shape[2], x_0.shape[3]))
            full_path_noise = torch.empty((self.num_steps, x_0.shape[0], x_0.shape[1], x_0.shape[2], x_0.shape[3]))
            full_path[0] = x_0

            for step in range(self.num_steps):
                t_idx = torch.ones((x_0.shape[0], )).long().to(device) * (step)
                isotropic_noise = torch.randn_like(x_0)

                x_t = (
                    extract(self.sqrt_alphas_bar, t_idx, x_0.shape) * x_0 +
                    extract(self.sqrt_one_minus_alphas_bar, t_idx, x_0.shape) * isotropic_noise
                    )
                full_path[step+1] = x_t
                full_path_noise[step] = extract(self.sqrt_one_minus_alphas_bar, t_idx, x_0.shape) * isotropic_noise

            x_T = x_t
            save_video(sample_dir, full_path_noise.cpu(), "forward_process_noise_masks.mp4")
            save_video(sample_dir, full_path.cpu(), "forward_process_prior_x_T.mp4")
            save_png(sample_dir, x_T.cpu(), "forward_diffusion_prior_x_T.png")

        else:
            T_idx = torch.ones((x_0.shape[0], )).long().to(device) * (self.num_steps - 1)
            isotropic_noise = torch.randn_like(x_0)

            x_T = (
                extract(self.sqrt_alphas_bar, T_idx, x_0.shape) * x_0 +
                extract(self.sqrt_one_minus_alphas_bar, t_idx, x_0.shape) * isotropic_noise
                )
            
        return x_T
    
    def forward(self, x_0):
        """
        Loss computation diffusion routine
        """
        t_indices = torch.randint(self.num_steps, size=(x_0.shape[0], ), device=x_0.device).long()
        isotropic_noise = torch.randn_like(x_0)

        x_t = (
            extract(self.sqrt_alphas_bar, t_indices, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t_indices, x_0.shape) * isotropic_noise
            )
        
        losses = F.mse_loss(self.model(x_t.float(), t_indices.float()), isotropic_noise, reduction='none')
        batch_losses = torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)

        return torch.mean(batch_losses)

class ddpm_sampler(nn.Module):
    def __init__(self, config, model, device):

        super().__init__()

        self.model = model
        self.num_steps = config.diffusion.num_steps
        self.register_buffer('betas', torch.linspace(config.diffusion.beta_min, config.diffusion.beta_max, self.num_steps).double().to(device))

        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1,0], value=1)[:self.num_steps]

        self.register_buffer('alphas_bar', alphas_bar.to(device))
        self.register_buffer('alphas_bar_prev', alphas_bar_prev.to(device))
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar).to(device))
        self.register_buffer('sqrt_alphas_bar_prev', torch.sqrt(alphas_bar_prev).to(device))
        self.register_buffer('sqrt_one_min_alphas_bar', torch.sqrt(1.0 - alphas_bar).to(device))
        self.register_buffer('sqrt_one_min_alphas_bar_prev', torch.sqrt(1.0 - alphas_bar_prev).to(device))
        
        # Needed to compute q(x_t | x_{t-1})
        # ===================================
        self.register_buffer('sqrt_recip_alphas_bar', torch.sqrt(1.0 / alphas_bar).to(device))
        self.register_buffer('sqrt_recip_alphas_bar_min_1', torch.sqrt(1.0 / alphas_bar - 1.0).to(device))

    # \alpha_{t|s}
    def alpha_t_over_s(self, t_idx, xt_shape):
        return (
            extract(self.sqrt_alphas_bar, t_idx, xt_shape) /
            extract(self.sqrt_alphas_bar_prev, t_idx, xt_shape)
        )
    
    # {\alpha_{t|s}}^2
    def alpha_t_over_s_sqrd(self, t_idx, xt_shape):
        return (
            extract(self.alphas_bar, t_idx, xt_shape) /
            extract(self.alphas_bar_prev, t_idx, xt_shape)
        )
    
    # {\sigma_t}^2
    def sigma_t_sqrd(self, t_idx, xt_shape):
        return (
                (1.0 - extract(self.alphas_bar, t_idx, xt_shape))
               )
    # {\sigma_s}^2
    def sigma_s_sqrd(self, t_idx, xt_shape):
        return (
                (1.0 - extract(self.alphas_bar_prev, t_idx, xt_shape))
               )
    # {\sigma_{t|s}}^2
    def sigma_t_over_s_sqrd(self, t_idx, xt_shape):
        return (
                (self.sigma_t_sqrd(t_idx, xt_shape)) - (self.alpha_t_over_s(t_idx, xt_shape) * self.sigma_s_sqrd(t_idx, xt_shape))
                )
    
    # Eq. 8
    def posterior_mean(self, x_t, pred_x0, posterior_var, t_idx):    
        alpha_t_bar = extract(self.alphas_bar, t_idx, x_t.shape)
        alpha_s_bar = extract(self.alphas_bar_prev,t_idx, x_t.shape)
        sqrt_alpha_t_bar = extract(self.sqrt_alphas_bar, t_idx, x_t.shape)
        sqrt_alpha_s_bar = extract(self.sqrt_alphas_bar_prev, t_idx, x_t.shape)

        first_term = ((sqrt_alpha_t_bar / sqrt_alpha_s_bar) / (1.0 - (alpha_t_bar / alpha_s_bar))) * x_t
        second_term = (sqrt_alpha_s_bar / (1.0 - alpha_s_bar)) * pred_x0

        return posterior_var * (first_term + second_term)

    # Eq. 7
    def posterior_variance(self, x_t_shape, t_idx):
        alpha_t_bar = extract(self.alphas_bar, t_idx, x_t_shape)
        alpha_s_bar = extract(self.alphas_bar_prev,t_idx, x_t_shape)

        first_term = (1.0 / (1.0 - alpha_s_bar))
        second_term = ((alpha_t_bar / alpha_s_bar) / (1.0 - (alpha_t_bar / alpha_s_bar)))

        return torch.reciprocal(first_term + second_term)

    def predict_x_0_from_epsilon(self, x_t, t_idx, eps):
        return (extract(self.sqrt_recip_alphas_bar, t_idx, x_t.shape) * x_t -
                extract(self.sqrt_recip_alphas_bar, t_idx, x_t.shape) * extract(self.sqrt_one_min_alphas_bar, t_idx, x_t.shape) * eps )  

    
    @torch.no_grad()   
    def sample_deterministically(self, x_T, sampling_noises):

        assert sampling_noises.shape[0] == self.num_steps, "`sampling_noises` should be a tensor of size (n,c,w,h), with n the number of diffusion steps"
        """
        Sampling routine.
        """
        x_t = x_T
        full_path = torch.empty((self.num_steps + 1, x_t.shape[0], x_t.shape[1], x_t.shape[2], x_t.shape[3]))
        full_path_x0 = torch.empty((self.num_steps + 1, x_t.shape[0], x_t.shape[1], x_t.shape[2], x_t.shape[3]))
        full_path_denoise_masks = torch.empty((self.num_steps - 1, x_t.shape[0], x_t.shape[1], x_t.shape[2], x_t.shape[3]))
        full_path[0] = x_t

        for step in reversed(range(1, self.num_steps)):
            t_indices = torch.ones((x_t.shape[0], ), device=x_t.device).long() * step
            pred_eps = self.model(x_t.float(), t_indices.float())
            pred_x0 = self.predict_x_0_from_epsilon(x_t, t_indices, pred_eps)

            # Compute predicted posterior variance and mean
            var = self.posterior_variance(x_t.shape, t_indices)
            mu = self.posterior_mean(x_t, pred_x0, var, t_indices)

            # Update the state x_t --> x_{t-1}
            if step > 0: 
                noise = sampling_noises[step]
            else:
                noise = 0

            x_t = mu + torch.sqrt(var) * noise
            full_path[self.num_steps - step] = x_t
            full_path_x0[self.num_steps - step] = pred_x0
            full_path_denoise_masks[self.num_steps - step - 1] = pred_eps

        x_0 = x_t
        return torch.clip(x_0, -1, 1), full_path, full_path_denoise_masks, full_path_x0
    
    @torch.no_grad()   
    def forward(self, x_T):
        """
        Sampling routine.
        """
        x_t = x_T
        full_path = torch.empty((self.num_steps + 1, x_t.shape[0], x_t.shape[1], x_t.shape[2], x_t.shape[3]))
        full_path_denoise_masks = torch.empty((self.num_steps - 1, x_t.shape[0], x_t.shape[1], x_t.shape[2], x_t.shape[3]))
        full_path[0] = x_t
    
        for step in reversed(range(1, self.num_steps)):
            t_indices = torch.ones((x_t.shape[0], ), device=x_t.device).long() * step
            pred_eps = self.model(x_t.float(), t_indices.float())
            pred_x0 = self.predict_x_0_from_epsilon(x_t, t_indices, pred_eps)

            # Compute predicted posterior variance and mean
            var = self.posterior_variance(x_t.shape, t_indices)
            mu = self.posterior_mean(x_t, pred_x0, var, t_indices)

            # Update the state x_t --> x_{t-1}
            if step > 0: 
                noise = torch.from_numpy(np.random.randn(*x_t.shape)).to(x_t.device)
            else:
                noise = 0

            x_t = mu + torch.sqrt(var) * noise
            full_path[self.num_steps - step] = x_t
            full_path_denoise_masks[self.num_steps - step - 1] = pred_eps

        x_0 = x_t
        return torch.clip(x_0, -1, 1), full_path, full_path_denoise_masks
