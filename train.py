import torch
from tqdm import tqdm
import wandb
from torchvision.utils import make_grid
from math import sqrt
from diffusion import DiffusionWrapper, EWMAWrapper, UNetConfig, UNet
import numpy as np
import logging

from dataclasses import dataclass, field
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from contextlib import contextmanager
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def cycle(loader):
    while True:
        for data in loader:
            yield data


@dataclass
class StepTimer:
    times: list = field(default_factory=list)

    @contextmanager
    def record(self):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        self.times.append((start, end))

    def sync_and_report(self):
        torch.cuda.synchronize()
        total = sum(s.elapsed_time(e) for s, e in self.times) / 1e3
        mean = total / len(self.times) if self.times else 0.0
        self.times.clear()
        return total, mean


class NormalizeTransform(torch.nn.Module):
    def forward(self, img):
        return (img * 2.0) - 1.0


def unnormalize(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) / 2.0


def create_dataset():
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            NormalizeTransform(),
        ]
    )
    valid_transform = transforms.Compose([transforms.ToTensor(), NormalizeTransform()])

    train_dataset = CIFAR10(
        root="./data",
        train=True,
        transform=train_transform,
        download=True,
    )

    val_dataset = CIFAR10(
        root="./data",
        train=False,
        transform=valid_transform,
        download=True,
    )

    return train_dataset, val_dataset


def save_checkpoint(model: torch.nn.Module, step: int, config: dict, filename: str):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "step": step,
        "config": config,
    }
    torch.save(checkpoint, filename)


def prepare_grid(samples: torch.Tensor, dims: tuple[int, int, int], n_samples: int) -> np.ndarray:
    C, H, W = dims
    samples = samples.reshape(-1, C, H, W)

    samples = unnormalize(samples)

    grid = make_grid(samples, nrow=int(sqrt(n_samples)))
    return grid.permute(1, 2, 0).cpu().detach().numpy()


def prepare_cuda_graph(
    ddpm: DiffusionWrapper,
    model: UNet,
    ema_ddpm: DiffusionWrapper,
    static_batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_acc: torch.Tensor,
):
    model.train()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(s):
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = ddpm.forward_batch(static_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            ema_ddpm.model.update(ddpm.model)  # type: ignore

    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = ddpm.forward_batch(static_batch)

        loss_acc.add_(loss.detach())

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        ema_ddpm.model.update(ddpm.model)  # type: ignore

    return g


def generate_samples(
    n_samples: int,
    ddpm: DiffusionWrapper,
    dims: tuple[int, int, int],
) -> tuple[np.ndarray, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()
    samples = ddpm.generate_samples(n_samples=n_samples)
    end.record()
    torch.cuda.synchronize()

    gen_time_s = start.elapsed_time(end) / 1000.0
    logger.info(f"generate_samples took {gen_time_s:.2f}s")

    grid = prepare_grid(samples, dims, n_samples)
    return grid, gen_time_s


def generate_and_validate(
    model: DiffusionWrapper,
    val_loader,
    validation_steps: int,
    n_samples: int,
    gen_samples: bool = False,
) -> dict:
    metrics_dict = {}
    if gen_samples:
        grid, gen_time = generate_samples(n_samples, model, (3, 32, 32))

        images = wandb.Image(grid)

        metrics_dict[f"images/images_{model.max_t}"] = images  # type: ignore
        metrics_dict["train/gen_time_s"] = gen_time  # type: ignore

    running_loss = torch.tensor(0.0, device=model.device)
    for _ in tqdm(range(validation_steps), total=validation_steps):
        batch, _ = next(val_loader)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model.forward_batch(batch)
                running_loss += loss

    metrics_dict["valid/loss"] = running_loss.sum().item() / validation_steps  # type: ignore

    return metrics_dict


def main():
    max_t = 1000
    max_lr = 1e-4

    val_sample_freq = 390  # around once an epoch
    val_gen_freq = 10 * 390  # around once every 10 epochs
    val_steps = 25

    n_samples = 64
    schedule_type = "cosine"
    n_train_steps = 100001
    batch_size = 128

    timer = StepTimer()
    cumtime = 0.0

    generate_sample = True

    model = UNet(
        UNetConfig(
            3,
            base_hidden=128,
            scales=[1, 2, 2, 2],  # 32M param
            n_heads=4,
            p_drop=0.2,
            attn_resolutions=[16, 8],
            in_resolution=32,
            n_resblocks=2,
            fp8_convs_enabled=True,
            fp8_scaling_type="delayed",
        )
    )

    config_dict = model.config.as_dict()
    config_dict["hparams.max_t"] = max_t
    config_dict["hparams.max_lr"] = max_lr
    config_dict["hparams.val_sample_freq"] = val_sample_freq
    config_dict["hparams.val_steps"] = val_steps
    config_dict["hparams.n_samples"] = n_samples
    config_dict["hparams.schedule_type"] = schedule_type
    config_dict["hparams.n_train_steps"] = n_train_steps
    config_dict["hparams.batch_size"] = batch_size
    config_dict["hparams.model_size"] = sum(p.numel() for p in model.parameters())

    logger.info(f"{config_dict=}")

    checkpoint_filename = f"checkpoint_{schedule_type}.pt"

    model = model.to(memory_format=torch.channels_last)  # type: ignore
    model.cuda()
    model: UNet = torch.compile(model)  # type: ignore
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, fused=True, capturable=True)
    ddpm = DiffusionWrapper(model, schedule_type=schedule_type, max_t=max_t, in_dim=32)
    ema_ddpm = DiffusionWrapper(
        EWMAWrapper(model, beta=0.9999, compile=True),
        schedule_type=schedule_type,
        max_t=max_t,
        in_dim=32,
    )

    assert isinstance(ema_ddpm.model, EWMAWrapper)
    assert isinstance(ddpm.model, torch.nn.Module)

    _ = wandb.init(project="diffusion-exps")
    wandb.config.update(config_dict)

    train_dataset, valid_dataset = create_dataset()

    train_loader = cycle(
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
    )

    val_loader = cycle(
        DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
    )

    static_batch = torch.zeros(
        batch_size,3,32,32,device="cuda",
    ).to(memory_format=torch.channels_last)
    loss_acc = torch.zeros((), device="cuda")

    graph = prepare_cuda_graph(ddpm, model, ema_ddpm, static_batch, optimizer, loss_acc)

    for step in tqdm(range(n_train_steps), total=n_train_steps):
        batch, _ = next(train_loader)

        if step >= n_train_steps:
            break

        static_batch.copy_(batch)

        with timer.record():
            graph.replay()

        if step % val_sample_freq == 0:
            cumtime_epoch, mean_time_step = timer.sync_and_report()
            cumtime += cumtime_epoch

            loss_dict = {}

            steps_since_log = val_sample_freq if step > 0 else 1
            avg_loss = loss_acc.item() / steps_since_log
            loss_acc.zero_()
            loss_dict["train/loss"] = avg_loss
            loss_dict["train/cumtime"] = cumtime
            loss_dict["train/mean_time_per_batch_s"] = mean_time_step

            save_checkpoint(ema_ddpm.model.model, step, config_dict, checkpoint_filename)

            do_gen_samples = generate_sample and (step % val_gen_freq == 0)

            metrics = generate_and_validate(
                ema_ddpm, val_loader, val_steps, n_samples, do_gen_samples
            )

            loss_dict |= metrics

            wandb.log(loss_dict)

    loss_dict = {}

    steps_since_log = val_sample_freq if step > 0 else 1
    avg_loss = loss_acc.item() / steps_since_log
    loss_acc.zero_()
    loss_dict["train/loss"] = avg_loss

    save_checkpoint(ema_ddpm.model.model, step, config_dict, checkpoint_filename)
    do_gen_samples = True

    metrics = generate_and_validate(ema_ddpm, val_loader, val_steps, n_samples, do_gen_samples)
    loss_dict |= metrics
    wandb.log(loss_dict)


if __name__ == "__main__":
    main()
