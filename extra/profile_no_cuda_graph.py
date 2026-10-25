import torch
from tqdm import tqdm
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from extra.arithmetic_intensity import unet_counts
from diffusion.unet import UNetConfig, UNet
from diffusion.ddpm import DiffusionWrapper, EWMAWrapper
import logging
from torch.profiler import record_function


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def cycle(loader):
    while True:
        for data in loader:
            yield data


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


def train_step(
    ddpm: DiffusionWrapper,
    model: UNet,
    ema_ddpm: DiffusionWrapper,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
):
    model.train()

    with record_function("train-forward"):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = ddpm.forward_batch(batch)

    with record_function("train-backward"):
        loss.backward()

    with record_function("grad-clip"):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    with record_function("optimizer-step"):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    with record_function("ema-update"):
        ema_ddpm.model.update(ddpm.model)  # type: ignore


def main():
    torch.manual_seed(3601627)
    max_t = 1000
    max_lr = 1e-4

    schedule_type = "linear"
    n_train_steps = 55
    batch_size = 128

    compile = True

    train_dataset, _ = create_dataset()

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
    model_config = model.config

    model = model.to(memory_format=torch.channels_last)  # type: ignore

    config_dict = model.config.as_dict()
    config_dict["hparams.max_t"] = max_t
    config_dict["hparams.max_lr"] = max_lr
    config_dict["hparams.schedule_type"] = schedule_type
    config_dict["hparams.n_train_steps"] = n_train_steps
    config_dict["hparams.batch_size"] = batch_size
    config_dict["hparams.model_size"] = sum(p.numel() for p in model.parameters())

    logger.info(f"{config_dict=}")
    model.cuda()

    if compile:
        model: UNet = torch.compile(model)  # type: ignore

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, fused=True)

    ddpm = DiffusionWrapper(model, schedule_type=schedule_type, max_t=max_t, in_dim=32)

    ema_ddpm = DiffusionWrapper(
        EWMAWrapper(model, beta=0.9999, compile=compile),
        schedule_type=schedule_type,
        max_t=max_t,
        in_dim=32,
    )

    assert isinstance(ema_ddpm.model, EWMAWrapper)
    assert isinstance(ddpm.model, torch.nn.Module)

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

    do_tb_profile = False  # if we want detailed profiler stats

    if do_tb_profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(skip_first=5, wait=0, warmup=45, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                "./log/prof", worker_name="bf16-new"
            ),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            for _ in tqdm(range(n_train_steps), total=n_train_steps):
                with record_function("get-data"):
                    batch, _ = next(train_loader)
                    batch = batch.to("cuda")
                    batch = batch.to(memory_format=torch.channels_last)

                model.train()
                with record_function("train-forward"):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = ddpm.forward_batch(batch)

                with record_function("train-backward"):
                    loss.backward()

                with record_function("grad-clip"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                with record_function("optimizer-step"):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                with record_function("ema-update"):
                    ema_ddpm.model.update(ddpm.model)

                prof.step()

            print(
                prof.key_averages(group_by_input_shape=True).table(
                    sort_by="cpu_time_total", row_limit=15
                )
            )
            print("\n" + "-" * 50)
            print("To view the detailed trace, run the following command in your terminal:")
            print("uv run tensorboard --logdir=./log/speedy")
            print("-" * 50)
            return

    else:
        warmup_steps = 5
        step_times = []
        model_times = []
        model_times_fwd = []

        count_parts = unet_counts(
            B=batch_size,
            in_ch=model_config.in_channels,
            C=model_config.base_hidden,
            scales=model_config.scales,
            nh=model_config.n_heads,
            attn_res=model_config.attn_resolutions,
            res0=model_config.in_resolution,
            n_resblocks=model_config.n_resblocks,
            p_drop=model_config.p_drop,
            elt=torch.empty((), dtype=torch.bfloat16).element_size(),
        )
        model_flops = sum(c.flops_fwd + c.flops_bwd for c in count_parts.values())
        model_bytes = sum(c.bytes_fwd + c.bytes_bwd for c in count_parts.values())

        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        model_start = torch.cuda.Event(True)
        model_end = torch.cuda.Event(True)

        model_start_fwd = torch.cuda.Event(True)
        model_end_fwd = torch.cuda.Event(True)

        torch.cuda.synchronize()
        for _ in tqdm(range(n_train_steps), total=n_train_steps):
            start.record()

            with record_function("get-data"):
                batch, _ = next(train_loader)
                batch = batch.to("cuda")
                batch = batch.to(memory_format=torch.channels_last)

            model.train()

            model_start.record()
            model_start_fwd.record()
            with record_function("train-forward"):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = ddpm.forward_batch(batch)
            model_end_fwd.record()

            with record_function("train-backward"):
                loss.backward()
            model_end.record()

            with record_function("grad-clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            with record_function("optimizer-step"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            with record_function("ema-update"):
                ema_ddpm.model.update(ddpm.model)

            end.record()
            end.synchronize()
            step_times.append(start.elapsed_time(end))
            model_times.append(model_start.elapsed_time(model_end))
            model_times_fwd.append(model_start_fwd.elapsed_time(model_end_fwd))

        kept = step_times[warmup_steps:]
        model_kept = model_times[warmup_steps:]
        model_fwd_kept = model_times_fwd[warmup_steps:]

        mean_step_ms = sum(kept) / len(kept)
        mean_model_ms = sum(model_kept) / len(model_kept)
        mean_model_fwd_ms = sum(model_fwd_kept) / len(model_fwd_kept)
        effective_tflops = model_flops / (mean_model_ms * 1e-3) / 1e12
        effective_bandwidth = model_bytes / (mean_model_ms * 1e-3) / 1e9
        best_tflops = model_flops / (min(model_kept) * 1e-3) / 1e12
        best_bandwidth = model_bytes / (min(model_kept) * 1e-3) / 1e9
        arithmetic_intensity = model_flops / model_bytes

        print(
            f"per-step (excl. {warmup_steps} warmup): "
            f"mean={mean_step_ms:.2f}ms  "
            f"min={min(kept):.2f}ms  max={max(kept):.2f}ms  "
            f"total={sum(kept):.2f}ms over {len(kept)} steps"
        )
        print(
            f"fwd window (UNet): mean={mean_model_fwd_ms:.2f}ms  "
            f"min={min(model_fwd_kept):.2f}ms  max={max(model_fwd_kept):.2f}ms"
        )
        print(
            f"fwd+bwd window (UNet + diffusion/loss): mean={mean_model_ms:.2f}ms  "
            f"min={min(model_kept):.2f}ms  max={max(model_kept):.2f}ms"
        )
        print(
            f"UNet logical work: {model_flops / 1e12:.3f} TFLOP, "
            f"{model_bytes / 1e9:.3f} GB, AI={arithmetic_intensity:.1f} FLOP/byte"
        )
        print(
            f"effective UNet (mean): {effective_tflops:.2f} TFLOP/s, "
            f"{effective_bandwidth:.1f} GB/s logical bandwidth"
        )
        print(
            f"effective UNet (best): {best_tflops:.2f} TFLOP/s, "
            f"{best_bandwidth:.1f} GB/s logical bandwidth"
        )


if __name__ == "__main__":
    main()
