"""Check the loaded NCCL library and optionally stress H3's async all-to-all path."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def runtime_nccl() -> tuple[int, Path]:
    paths = {
        Path(fields[5].strip()).resolve()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if len(fields := line.split(maxsplit=5)) == 6 and "libnccl.so" in fields[5]
    }
    if len(paths) != 1:
        raise RuntimeError(f"expected one loaded NCCL library, found {sorted(map(str, paths))}")
    path = paths.pop()
    library = ctypes.CDLL(str(path))
    function = library.ncclGetVersion
    function.argtypes = [ctypes.POINTER(ctypes.c_int)]
    function.restype = ctypes.c_int
    version = ctypes.c_int()
    status = function(ctypes.byref(version))
    if status:
        raise RuntimeError(f"ncclGetVersion failed with status {status}")
    return version.value, path


def _check(expected_version: str):
    version, path = runtime_nccl()
    expected = tuple(int(part) for part in expected_version.split("."))
    actual = (version // 10000, version % 10000 // 100, version % 100)
    wheel_path = (
        Path(torch.__file__).resolve().parent.parent / "nvidia/nccl/lib/libnccl.so.2"
    ).resolve()
    if actual != expected or path != wheel_path:
        raise RuntimeError(
            f"expected wheel NCCL {expected_version} at {wheel_path}, got {actual} at {path}"
        )
    return {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "compiled_nccl": torch.cuda.nccl.version(), "runtime_nccl": actual,
        "library": str(path),
    }


def _stress(iterations: int, expected_version: str):
    world, rank, local = (int(os.environ[key]) for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK"))
    if world != 8:
        raise ValueError("the H3 stress test requires exactly eight GPU ranks")
    torch.cuda.set_device(local)
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", local), timeout=timedelta(seconds=90),
    )
    try:
        report = _check(expected_version)
        if rank == 0:
            print(json.dumps(report), flush=True)
        destinations = torch.arange(world, device="cuda") * world + rank
        payloads = [
            (destinations + lane).to(torch.int8)[:, None, None].expand(world, 512, 128).clone()
            for lane in range(3)
        ]
        scales = [
            (destinations + lane + 1).float()[:, None].expand(world, 512).clone()
            for lane in range(3)
        ]
        pinned = torch.zeros(8 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
        destination = torch.empty_like(pinned, device="cuda")
        transfer = torch.cuda.Stream()
        for iteration in range(iterations):
            outputs = [torch.empty_like(tensor) for tensor in payloads]
            output_scales = [torch.empty_like(tensor) for tensor in scales]
            pending = []
            for payload, scale, output, output_scale in zip(payloads, scales, outputs, output_scales):
                pending.append(dist.all_to_all_single(output, payload, async_op=True))
                pending.append(dist.all_to_all_single(output_scale, scale, async_op=True))
            with torch.cuda.stream(transfer):
                destination.copy_(pinned, non_blocking=True)
            for work in pending:
                work.wait()
            torch.cuda.synchronize()
            for lane in range(3):
                for source in range(world):
                    expected = source + rank * world + lane
                    if not bool((outputs[lane][source] == expected).all()) or not bool(
                        (output_scales[lane][source] == expected + 1).all()
                    ):
                        raise AssertionError(f"incorrect all-to-all result at iteration {iteration}")
        dist.barrier(device_ids=[local])
        if rank == 0:
            print(f"PASS: {iterations} async INT8/FP32 all-to-all iterations with pinned H2D", flush=True)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", default="2.28.9")
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.stress:
        _stress(args.iterations, args.expected_version)
    else:
        print(json.dumps(_check(args.expected_version), indent=2))


if __name__ == "__main__":
    main()
