#!/usr/bin/env python3
"""Hold host RAM and optionally GPU memory for EMNLP training allocation."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time


STOP = False


def _handle_stop(signum: int, _frame: object) -> None:
    global STOP
    STOP = True
    print(f"Received signal {signum}; exiting after current sleep.", flush=True)


def _reserve_ram(gib: int, stride_bytes: int) -> bytearray:
    if gib < 0:
        raise ValueError("--ram-gib must be non-negative")
    if stride_bytes <= 0:
        raise ValueError("--touch-stride-bytes must be positive")

    total_bytes = gib * 1024**3
    buf = bytearray(total_bytes)
    for i in range(0, total_bytes, stride_bytes):
        buf[i] = 1
    return buf


def _parse_device_list(raw: str | None, count: int) -> list[int]:
    if not raw:
        return list(range(count))

    devices: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        device = int(part)
        if device < 0 or device >= count:
            raise ValueError(f"GPU device index {device} is outside visible range 0..{count - 1}")
        devices.append(device)
    return devices


def _reserve_gpu_memory(gib_per_gpu: float, devices_raw: str | None, chunk_mib: int) -> list[object]:
    if gib_per_gpu <= 0:
        return []
    if chunk_mib <= 0:
        raise ValueError("--gpu-chunk-mib must be positive")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("GPU memory reservation requires PyTorch in the environment.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch.")

    visible_count = torch.cuda.device_count()
    devices = _parse_device_list(devices_raw, visible_count)
    if not devices:
        return []

    total_bytes = int(gib_per_gpu * 1024**3)
    chunk_bytes = chunk_mib * 1024**2
    tensors: list[object] = []

    for device in devices:
        allocated = 0
        with torch.cuda.device(device):
            while allocated < total_bytes:
                this_chunk = min(chunk_bytes, total_bytes - allocated)
                tensor = torch.empty((this_chunk,), dtype=torch.uint8, device=f"cuda:{device}")
                tensor.fill_(1)
                tensors.append(tensor)
                allocated += this_chunk
            torch.cuda.synchronize(device)
        print(
            f"Reserved ~{gib_per_gpu:g} GiB GPU memory on cuda:{device} "
            f"({torch.cuda.get_device_name(device)})",
            flush=True,
        )

    return tensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reserve host/container RAM, optionally reserve GPU memory, and keep the process alive."
    )
    parser.add_argument(
        "--ram-gib",
        type=int,
        default=40,
        help="RAM to reserve in GiB by allocating and touching a bytearray.",
    )
    parser.add_argument(
        "--touch-stride-bytes",
        type=int,
        default=4096,
        help="Page-touch stride used after allocation.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=int,
        default=0,
        help="Seconds to hold resources. Use 0 to hold until interrupted.",
    )
    parser.add_argument(
        "--gpu-mem-gib-per-gpu",
        type=float,
        default=0.0,
        help="GPU memory to reserve on each selected visible CUDA device. Default: 0.",
    )
    parser.add_argument(
        "--gpu-devices",
        default="",
        help="Comma-separated visible CUDA device indices to reserve. Default: all visible devices.",
    )
    parser.add_argument(
        "--gpu-chunk-mib",
        type=int,
        default=1024,
        help="Chunk size for GPU allocations in MiB.",
    )
    return parser.parse_args()


def main() -> int:
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _handle_stop)

    args = parse_args()

    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    print("CUDA_GPU_DEVICES =", os.environ.get("CUDA_GPU_DEVICES"), flush=True)
    print(f"Reserving ~{args.ram_gib} GiB RAM", flush=True)

    try:
        buf = _reserve_ram(args.ram_gib, args.touch_stride_bytes)
    except MemoryError:
        print(f"Failed to reserve {args.ram_gib} GiB RAM.", file=sys.stderr, flush=True)
        return 1

    print(f"Reserved ~{args.ram_gib} GiB RAM", flush=True)

    try:
        gpu_tensors = _reserve_gpu_memory(
            args.gpu_mem_gib_per_gpu,
            args.gpu_devices,
            args.gpu_chunk_mib,
        )
    except Exception as exc:
        print(f"Failed to reserve GPU memory: {exc}", file=sys.stderr, flush=True)
        return 1

    started = time.monotonic()
    while not STOP:
        if args.hold_seconds > 0:
            remaining = args.hold_seconds - (time.monotonic() - started)
            if remaining <= 0:
                break
            time.sleep(min(60, remaining))
        else:
            time.sleep(60)

    # Keep the allocation live until the hold loop has finished.
    del gpu_tensors
    del buf
    print("Released reserved resources.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
