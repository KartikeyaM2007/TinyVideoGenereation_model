import torch
import torch.nn.functional as F


class GaussianDiffusion2D:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.timesteps = timesteps
        self.device = torch.device(device)

        betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def _extract(self, arr, t, x_shape):
        out = arr.gather(0, t)
        return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_ab = self._extract(self.sqrt_alpha_bars, t, x_start.shape)
        sqrt_omab = self._extract(self.sqrt_one_minus_alpha_bars, t, x_start.shape)
        return sqrt_ab * x_start + sqrt_omab * noise

    def p_losses(self, model, x_start, t, y):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        noise_pred = model(x_noisy, t, y)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def predict_eps(self, model, x, t, y, guidance_scale=1.0):
        if guidance_scale <= 1.0:
            return model(x, t, y)

        eps_cond = model(x, t, y)
        eps_uncond = model(x, t, torch.zeros_like(y))
        return eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    @torch.no_grad()
    def sample(self, model, shape, y, sampling_steps=100, guidance_scale=1.0):
        model.eval()
        x = torch.randn(shape, device=self.device)

        if sampling_steps >= self.timesteps:
            steps = torch.arange(self.timesteps - 1, -1, -1, device=self.device).long()
        else:
            steps = torch.linspace(self.timesteps - 1, 0, sampling_steps, device=self.device).long()

        for i, step in enumerate(steps):
            t = torch.full((shape[0],), int(step.item()), device=self.device, dtype=torch.long)
            eps = self.predict_eps(model, x, t, y, guidance_scale=guidance_scale)

            alpha_bar_t = self._extract(self.alpha_bars, t, x.shape)
            x0_pred = (x - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            if i == len(steps) - 1:
                x = x0_pred
                break

            prev_step = steps[i + 1]
            prev_t = torch.full((shape[0],), int(prev_step.item()), device=self.device, dtype=torch.long)
            alpha_bar_prev = self._extract(self.alpha_bars, prev_t, x.shape)

            x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1.0 - alpha_bar_prev) * eps

        return x.clamp(-1.0, 1.0)