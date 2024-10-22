import torch 
import numpy as np
import decimal 
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import save_exr, save_png, save_video, compute_gradient

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

"""
Transition functions (\tau(t)).
"""
def exponential_transition_scheme(num_steps, t_idx, transition_pt, k):
    ts = torch.linspace(0.0, 1.0, num_steps).to(t_idx.device)
    ts = ts.gather(dim=0, index=t_idx)
    exp_scaling = 1.0 / (1.0 + torch.exp((-k * ts) + (transition_pt * k)))
    return exp_scaling

def linear_transition_scheme(num_steps, t_idx, transition_pt, k):
    ts = torch.linspace(0.0, 1.0, num_steps).to(t_idx.device)
    ts = ts.gather(dim=0, index=t_idx)
    return torch.where(ts < transition_pt, ts, torch.ones(ts.shape).to(t_idx.device))

def cosine_transition_scheme(num_steps, t_idx, transition_pt, k):
    ts = torch.linspace(0.0, 1.0, num_steps).to(t_idx.device)
    ts = ts.gather(dim=0, index=t_idx)
    cosine_scaling = torch.add(torch.cos(torch.add(ts * np.pi, np.pi)), 1.0) / 2.0
    return cosine_scaling


def square_of_sum(x, y):
    """
    Given x and y, computes (x + y)^2
    """
    return torch.square(x) + 2 * x * y + torch.square(y)


class edge_preserving_diffusion_trainer(nn.Module):
    def __init__(self, config, model, trans_fn, device):

        super().__init__()

        self.model = model
        self.lambda_min = config.diffusion.lambda_min
        self.lambda_max = config.diffusion.lambda_max
        self.transition_fn = trans_fn
        self.transition_pt = config.diffusion.transition_pt
        self.k = config.diffusion.k # Steepness of exponential trans fn (has no impact if other trans fn is used)
        self.num_steps = config.diffusion.num_steps

        self.register_buffer('betas', torch.linspace(config.diffusion.beta_min, config.diffusion.beta_max, self.num_steps).double().to(device))
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # =======================================
        # To compute q(x_t | x_{t-1}) and others
        # =======================================
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar).to(device))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar).to(device))

    # \lambda(t)    
    def time_idx_to_lambda(self, t_idx):
        time_fractions = t_idx.float() / (self.transition_pt * self.num_steps)
        lambdas = (1 - time_fractions) * self.lambda_min + time_fractions * self.lambda_max
        return lambdas

    # Non-isotropic variance according to [Perona and Malik, 1990] 
    def g_gradient(self, x, t_idx):
        x_grad = compute_gradient(x)
        one_min_alpha_bar_t = extract(self.sqrt_one_minus_alphas_bar, t_idx, x_grad.shape)
        lambdas = self.time_idx_to_lambda(t_idx)

        perona_malik = torch.sqrt(1.0 + (x_grad / lambdas[:, None, None, None]))
        time_transition_scaling = self.transition_fn(self.num_steps, t_idx, self.transition_pt, self.k)
        return one_min_alpha_bar_t / ((perona_malik * (1.0 - time_transition_scaling[:, None, None, None])) + time_transition_scaling[:, None, None, None])

    def sample_edge_preserving_noise(self, x, t_idx):
        """
        Compute anisotropic edge-preserving noise, given an image
        """
        g = self.g_gradient(x, t_idx)
        isotropic_noise = torch.randn_like(x)
        edge_preserving_noise = (isotropic_noise * g)
        return edge_preserving_noise
    
    def sample_edge_preserving_noise_deterministic(self, x, t_idx, iso_noise):
        """
        Compute anisotropic edge-preserving noise, given an image
        """
        g = self.g_gradient(x, t_idx)
        edge_preserving_noise = (iso_noise * g)
        return edge_preserving_noise
        
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
                aniso_noise = self.sample_edge_preserving_noise(x_0, t_idx)

                x_t = (
                    extract(self.sqrt_alphas_bar, t_idx, x_0.shape) * x_0 +
                    aniso_noise
                    )
                full_path[step+1] = x_t
                full_path_noise[step] = aniso_noise
            
            x_T = x_t
            save_video(sample_dir, full_path_noise.cpu(), "forward_process_noise_masks.mp4")
            save_video(sample_dir, full_path.cpu(), "forward_process_prior_x_T.mp4")
            save_png(sample_dir, x_T.cpu(), "forward_diffusion_prior_x_T.png")

        else:
            T_idx = torch.ones((x_0.shape[0], )).long().to(device) * (self.num_steps - 1)
            aniso_noise = self.sample_edge_preserving_noise(x_0, t_idx) 

            x_T = (
                extract(self.sqrt_alphas_bar, T_idx, x_0.shape) * x_0 +
                    aniso_noise 
                )
            
        return x_T
        
    def forward(self, x_0):
        """
        Loss computation
        """
        t_indices = torch.randint(self.num_steps, size=(x_0.shape[0], ), device=x_0.device).long()
        aniso_noise = self.sample_edge_preserving_noise(x_0, t_indices).float()

        x_t = extract(self.sqrt_alphas_bar, t_indices, x_0.shape) * x_0 + aniso_noise
        
        # Note that non-isotropic variance according to edges is explicitly learned
        losses = F.mse_loss(self.model(x_t.float(), t_indices.float()), aniso_noise, reduction='none')
        batch_losses = torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)

        return torch.mean(batch_losses)

class edge_preserving_sampler(nn.Module):
    # def __init__(self, beta_min, beta_max, lambda_min, lambda_max, T, dt, transition_fn, transition_pt, k, model, device):
    def __init__(self, config, model, trans_fn, device):
        super().__init__()

        self.model = model
        self.lambda_min = config.diffusion.lambda_min
        self.lambda_max = config.diffusion.lambda_max
        self.transition_fn = trans_fn
        self.transition_pt = config.diffusion.transition_pt
        self.k = config.diffusion.k # Steepness of exponential trans fn (has no impact if other trans fn is used)
        self.num_steps = config.diffusion.num_steps

        self.register_buffer('betas', torch.linspace(config.diffusion.beta_min, config.diffusion.beta_max, self.num_steps).double().to(device))

        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)   # \bar{\alpha_t}
        alphas_bar_prev = F.pad(alphas_bar, [1,0], value=1)[:self.num_steps]  # \bar{\alpha_s}

        self.register_buffer('alphas_bar', alphas_bar.to(device))
        self.register_buffer('alphas_bar_prev', alphas_bar_prev.to(device))
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar).to(device))
        self.register_buffer('sqrt_alphas_bar_prev', torch.sqrt(alphas_bar_prev).to(device))
        # ===================================
        # Needed to compute q(x_t | x_{t-1})
        # ===================================
        self.register_buffer('sqrt_recip_alphas_bar', torch.sqrt(1.0 / alphas_bar).to(device))
        self.register_buffer('sqrt_recip_alphas_bar_min_1', torch.sqrt(1.0 / alphas_bar - 1.0).to(device))

    # \lambda(t)    
    def time_idx_to_lambda(self, t_idx):
        time_fractions = t_idx.float() / (self.transition_pt * self.num_steps)
        lambdas = (1 - time_fractions) * self.lambda_min + time_fractions * self.lambda_max
        return lambdas

    # {\mathbf{sigma}_t}^2
    def sigma_t_sqrd(self, t_idx, grad_x0):
        tau_t = self.transition_fn(self.num_steps, t_idx, self.transition_pt, self.k)

        lambdas = self.time_idx_to_lambda(t_idx)

        return (
                (1.0 - extract(self.alphas_bar, t_idx, grad_x0.shape)) /
                square_of_sum((1.0 - tau_t)[:, None, None, None] * torch.sqrt(1.0 + (grad_x0 / lambdas[:, None, None, None])), tau_t[:, None, None, None])
               )
    
    # {\mathbf{sigma}_s}^2
    def sigma_s_sqrd(self, t_idx, grad_x0):
        s_idx = torch.maximum((t_idx - 1.0).long(), torch.zeros(t_idx.shape).to(t_idx.device).long()) # Handles case t_idx = 0
        tau_s = self.transition_fn(self.num_steps, s_idx , self.transition_pt, self.k)
        lambdas = self.time_idx_to_lambda(t_idx)

        return (
                (1.0 - extract(self.alphas_bar_prev, t_idx, grad_x0.shape)) /
                square_of_sum((1.0 - tau_s)[:, None, None, None] * torch.sqrt(1.0 + (grad_x0 / lambdas[:, None, None, None])), tau_s[:, None, None, None])
               )

    # \alpha_{t|s}
    def alpha_t_over_s(self, t_idx, grad_x0):
        return (
            extract(self.sqrt_alphas_bar, t_idx, grad_x0.shape) /
            extract(self.sqrt_alphas_bar_prev, t_idx, grad_x0.shape)
        )
    
    # \alpha_{t|s}^2
    def alpha_t_over_s_sqrd(self, t_idx, grad_x0):
        return (
            extract(self.alphas_bar, t_idx, grad_x0.shape) /
            extract(self.alphas_bar_prev, t_idx, grad_x0.shape)
        )

    # \mathbf{\sigma{t|s}^2}
    def sigma_t_over_s_sqrd(self, t_idx, grad_x0):
        return self.sigma_t_sqrd(t_idx, grad_x0) - (self.alpha_t_over_s_sqrd(t_idx, grad_x0) * self.sigma_s_sqrd(t_idx, grad_x0))

    # Eq. 16 
    def posterior_mean(self, x_t, pred_x0, posterior_var, grad_x0, t_idx):

        first_term = (
                        (self.alpha_t_over_s(t_idx, grad_x0) /
                        self.sigma_t_over_s_sqrd(t_idx, grad_x0)) * x_t
                    )

        alpha_s = extract(self.sqrt_alphas_bar_prev, t_idx, grad_x0.shape)
        second_term = (
                        (alpha_s  / 
                        self.sigma_s_sqrd(t_idx, grad_x0)) * pred_x0
                    )

        return posterior_var * (first_term + second_term)

    # Eq. 15
    def posterior_variance(self, grad_x0, t_idx):
        first_term = (
                        1.0 / 
                        self.sigma_s_sqrd(t_idx, grad_x0)
                    )
        second_term = ( 
                        self.alpha_t_over_s_sqrd(t_idx, grad_x0) / 
                        self.sigma_t_over_s_sqrd(t_idx, grad_x0)
                    )
        
        return torch.reciprocal(first_term + second_term)
     

    def predict_x_0_from_epsilon(self, x_t, t_idx, eps):
        return (extract(self.sqrt_recip_alphas_bar, t_idx, x_t.shape) * x_t -
                extract(self.sqrt_recip_alphas_bar, t_idx, x_t.shape) * eps )  
    
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

            # Predict x0 by first predicting the noise. Then compute the gradient image and transition scheme based on this prediction.
            pred_eps = self.model(x_t.float(), t_indices.float())
            pred_x0 = self.predict_x_0_from_epsilon(x_t, t_indices, pred_eps)
            grad_pred_x0 = compute_gradient(pred_x0)

            # Compute predicted posterior variance and mean
            var = self.posterior_variance(grad_pred_x0, t_indices)
            mu = self.posterior_mean(x_t,pred_x0, var, grad_pred_x0, t_indices)

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

            # Predict x0 by first predicting the noise. Then compute the gradient image and transition scheme based on this prediction.
            pred_eps = self.model(x_t.float(), t_indices.float())
            pred_x0 = self.predict_x_0_from_epsilon(x_t, t_indices, pred_eps)
            grad_pred_x0 = compute_gradient(pred_x0)

            # Compute predicted posterior variance and mean
            var = self.posterior_variance(grad_pred_x0, t_indices)
            mu = self.posterior_mean(x_t,pred_x0, var, grad_pred_x0, t_indices)

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
