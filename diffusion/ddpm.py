import torch
import torch.nn.functional as F
from math import pi
from tqdm import tqdm
from copy import deepcopy
from .unet import UNet
from torch.profiler import record_function


class EWMAWrapper:
    def __init__(self, model: UNet, beta: float = 0.9999, compile: bool = True):
        self.beta = beta
        self.model: UNet = deepcopy(model)
        self.model.eval()

        if compile:
            self.model: UNet = torch.compile(self.model, mode="reduce-overhead")  # type: ignore

    @torch.no_grad()
    @torch.compile
    def update(self, model: UNet | torch.nn.Module):
        for ema_v, model_v in zip(self.model.state_dict().values(), model.state_dict().values()):
            ema_v.mul_(self.beta)
            update = (1.0 - self.beta) * model_v
            ema_v.copy_(ema_v + update)

    @torch.no_grad()
    def __call__(self, *args, **kwds):
        return self.model(*args, **kwds)


class DiffusionWrapper:
    def __init__(
        self,
        model: UNet | EWMAWrapper,
        schedule_type: str,
        max_t: int,
        in_dim: int,
        device: str = "cuda",
        vlb_scale: float = 0.001,
    ):
        self.device = torch.device(device)

        alphas, alpha_bars, betas, q_vars, p_vars = self._make_schedule(max_t, schedule_type)

        self.alphas = alphas.to(self.device)
        self.alpha_bars = alpha_bars.to(self.device)
        self.betas = betas.to(self.device)
        self.q_vars = q_vars.to(self.device)
        self.p_vars = p_vars.to(self.device)

        # 1.0 / sqrt(a_t)
        self.recipsqrt_a = 1.0 / (self.alphas.sqrt())

        # beta_t / (sqrt(1 - a_bar_t))
        self.noise_coeff = self.betas / (1.0 - self.alpha_bars).sqrt()
        
        self.max_t = max_t
        self.in_dim = in_dim
        self.model = model

    @torch.no_grad()
    def _denoise_step(
        self,
        samples: torch.Tensor,
        t_val: int,
        recipsqrt_a_t: torch.Tensor,
        noise_coeff_t: torch.Tensor,
        beta_t: torch.Tensor,
        q_var_t: torch.Tensor,
    ) -> torch.Tensor:
        n_samples = samples.shape[0]
        t_tensor = torch.full((n_samples,), t_val, device=samples.device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            eps_pred = self.model(samples, t_tensor)

        # TODO: We need to .clamp(-1,1) here?
        samples = recipsqrt_a_t * (samples - noise_coeff_t * eps_pred)
        if t_val > 0:
            z = torch.randn_like(samples)
            samples += z * beta_t.sqrt()
        return samples

    @torch.no_grad()
    def generate_samples(
        self,
        n_samples: int,
    ) -> torch.Tensor:
        step_range = list(range(0, self.max_t))

        samples = torch.randn(
            size=(n_samples, 3, self.in_dim, self.in_dim),
            device=self.device,
        ).to(memory_format=torch.channels_last)

        recipsqrt_a = self.recipsqrt_a
        noise_coeff = self.noise_coeff
        betas = self.betas
        q_vars = self.q_vars

        step_range = list(reversed([(i, t) for i, t in enumerate(step_range)]))

        for idx, t in tqdm(
            (step_range),
            desc="Generating samples.",
            total=len(step_range),
        ):
            samples = self._denoise_step(
                samples,
                t,
                recipsqrt_a[idx][..., None, None, None],
                noise_coeff[idx][..., None, None, None],
                betas[idx][..., None, None, None],
                q_vars[idx][..., None, None, None],
            )

        return samples.clamp(-1.0, 1.0)

    def _make_schedule(self, max_t: int, schedule_type: str):
        match schedule_type:
            case "linear":
                scale = max_t / 1000
                betas = torch.linspace(
                    (10 ** (-4) / scale),
                    (0.02 / scale),
                    steps=max_t,
                    dtype=torch.float64,
                )
                alphas = 1.0 - betas
                alpha_bars = torch.cumprod(alphas, dim=-1)

            case "cosine":
                s = 0.008
                steps = max_t + 1
                t = torch.linspace(0, max_t, steps, dtype=torch.float64) / max_t
                alphas_cumprod = torch.cos((t + s) / (1 + s) * pi * 0.5) ** 2
                alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

                betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
                betas = torch.clip(betas, 0, 0.999)

                alphas = 1.0 - betas
                alpha_bars = torch.cumprod(alphas, dim=-1)

            case _:
                raise ValueError

        alpha_bars_prev = torch.concat(
            [torch.tensor(1.0, dtype=torch.float64)[None], alpha_bars[:-1]]
        )

        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)  # \tilde{\beta}

        q_posterior_variance = torch.concat(
            [
                posterior_variance[1:2],
                posterior_variance[1:],
            ]  # vars of q(x_{t-1}| x_t, x_0)
        )

        p_posterior_variance = torch.concat(
            [posterior_variance[1:2], betas[1:]]
        )  # vars of p(x_{t-1} | x_t)

        return (
            alphas.float(),
            alpha_bars.float(),
            betas.float(),
            q_posterior_variance.float(),
            p_posterior_variance.float(),
        )

    def noise_batch(
        self, x: torch.Tensor, t: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Prepares a batch for training.
        # Optionally supports the ability to create noised sample at a specific t

        if t is not None:
            eps = torch.randn_like(x, device=self.device)
            alpha_bar = self.alpha_bars[t][..., None, None, None]
            return (
                (alpha_bar.sqrt() * x) + ((1.0 - alpha_bar).sqrt() * eps),
                x,
                eps,
                torch.full((len(x),), device=self.device, fill_value=t),
            )

        else:
            assert t is None
            t_tens = torch.randint(
                low=0,
                high=(self.max_t),
                size=(len(x),),
                device=self.device,
            )
            eps = torch.randn_like(x, device=self.device)
            alpha_bar = self.alpha_bars[t_tens][..., None, None, None]

            return (
                (alpha_bar.sqrt() * x) + ((1.0 - alpha_bar).sqrt() * eps),
                x,
                eps,
                t_tens,
            )

    @torch.compile()
    def forward_batch(self, batch: torch.Tensor) -> torch.Tensor:
        with record_function("noise_batch"):
            batch, _, target, t = self.noise_batch(batch)
        out = self.model(batch, t)
        eps_pred = out
        loss_simple = F.mse_loss(eps_pred, target)
        return loss_simple