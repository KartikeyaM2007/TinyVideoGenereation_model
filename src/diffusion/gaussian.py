from __future__ import annotations

import torch


class GaussianDiffusion:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02, device=None):
        self.timesteps = timesteps
        self.device = device or torch.device("cpu")
        betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, x_shape):
        out = arr.gather(0, t)
        return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_ab = self._extract(self.sqrt_alpha_bars, t, x_start.shape)
        sqrt_omab = self._extract(self.sqrt_one_minus_alpha_bars, t, x_start.shape)
        return sqrt_ab * x_start + sqrt_omab * noise

    @torch.no_grad()
    def p_sample(self, model, x: torch.Tensor, t: torch.Tensor, text_emb: torch.Tensor, guidance_scale: float = 1.0):
        beta_t = self._extract(self.betas, t, x.shape)
        alpha_t = self._extract(self.alphas, t, x.shape)
        alpha_bar_t = self._extract(self.alpha_bars, t, x.shape)

        if guidance_scale != 1.0:
            eps_cond = model(x, t, text_emb)
            eps_uncond = model(x, t, torch.zeros_like(text_emb))
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = model(x, t, text_emb)

        mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(x.shape[0], *([1] * (len(x.shape) - 1)))
        return mean + nonzero_mask * torch.sqrt(beta_t) * noise

    @torch.no_grad()
    def sample(self, model, shape, text_emb: torch.Tensor, sampling_steps: int = 50, guidance_scale: float = 2.0):
        x = torch.randn(shape, device=self.device)
        steps = torch.linspace(self.timesteps - 1, 0, sampling_steps, device=self.device).long()
        for step in steps:
            t = torch.full((shape[0],), int(step.item()), device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t, text_emb, guidance_scale=guidance_scale)
        return x.clamp(-1, 1)
