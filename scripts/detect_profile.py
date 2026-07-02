#!/usr/bin/env python3
import argparse
import json
import os
import platform
import shlex
import subprocess


PROFILES = {
    "a100-80g": {
        "MAX_LENGTH": "8192",
        "BATCH_SIZE": "2",
        "GRAD_ACCUM": "4",
        "LR": "2e-4",
        "LORA_R": "32",
        "LORA_ALPHA": "64",
        "EPOCHS": "1",
        "SAMPLE_LIMIT": "0",
    },
    "a100-40g": {
        "MAX_LENGTH": "4096",
        "BATCH_SIZE": "2",
        "GRAD_ACCUM": "4",
        "LR": "2e-4",
        "LORA_R": "32",
        "LORA_ALPHA": "64",
        "EPOCHS": "1",
        "SAMPLE_LIMIT": "0",
    },
    "gpu8g": {
        "MAX_LENGTH": "1024",
        "BATCH_SIZE": "1",
        "GRAD_ACCUM": "8",
        "LR": "2e-4",
        "LORA_R": "16",
        "LORA_ALPHA": "32",
        "EPOCHS": "1",
        "SAMPLE_LIMIT": "0",
    },
    "gpu16g": {
        "MAX_LENGTH": "2048",
        "BATCH_SIZE": "1",
        "GRAD_ACCUM": "8",
        "LR": "2e-4",
        "LORA_R": "16",
        "LORA_ALPHA": "32",
        "EPOCHS": "1",
        "SAMPLE_LIMIT": "0",
    },
    "mps16g": {
        "MAX_LENGTH": "512",
        "BATCH_SIZE": "1",
        "GRAD_ACCUM": "8",
        "LR": "1e-4",
        "LORA_R": "8",
        "LORA_ALPHA": "16",
        "EPOCHS": "1",
        "SAMPLE_LIMIT": "0",
    },
    "cpu": {
        "MAX_LENGTH": "384",
        "BATCH_SIZE": "1",
        "GRAD_ACCUM": "16",
        "LR": "1e-4",
        "LORA_R": "4",
        "LORA_ALPHA": "8",
        "EPOCHS": "1",
        "SAMPLE_LIMIT": "0",
    },
}


def mac_memory_gb() -> float:
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return int(out) / 1000**3
    except Exception:
        return 0.0


def detect_auto() -> tuple[str, dict]:
    info = {
        "system": platform.system(),
        "machine": platform.machine(),
        "cuda": False,
        "mps": False,
        "device_name": "",
        "memory_gb": 0.0,
    }
    try:
        import torch

        if torch.cuda.is_available():
            idx = 0
            props = torch.cuda.get_device_properties(idx)
            info.update({
                "cuda": True,
                "device_name": props.name,
                "memory_gb": props.total_memory / 1000**3,
            })
            name = props.name.lower()
            if "a100" in name:
                if info["memory_gb"] >= 70:
                    return "a100-80g", info
                return "a100-40g", info
            if info["memory_gb"] >= 14:
                return "gpu16g", info
            return "gpu8g", info

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info.update({
                "mps": True,
                "device_name": "Apple MPS",
                "memory_gb": mac_memory_gb(),
            })
            return "mps16g", info
    except Exception as exc:
        info["torch_detection_error"] = str(exc)

    info["memory_gb"] = mac_memory_gb() if info["system"] == "Darwin" else 0.0
    return "cpu", info


def shell_defaults(profile: str, values: dict, info: dict) -> str:
    lines = [
        f"RESOLVED_PROFILE={shlex.quote(profile)}",
        f"HARDWARE_INFO={shlex.quote(json.dumps(info, ensure_ascii=False, sort_keys=True))}",
    ]
    for key, value in values.items():
        if key in os.environ:
            continue
        lines.append(f"{key}={shlex.quote(value)}")
    lines.append("export RESOLVED_PROFILE HARDWARE_INFO EPOCHS MAX_LENGTH BATCH_SIZE GRAD_ACCUM LR LORA_R LORA_ALPHA SAMPLE_LIMIT")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="auto", choices=["auto", "a100-40g", "a100-80g", "gpu8g", "gpu16g", "mps16g", "cpu"])
    parser.add_argument("--format", default="json", choices=["json", "shell"])
    args = parser.parse_args()

    if args.profile == "auto":
        profile, info = detect_auto()
    else:
        profile, info = args.profile, {"forced_profile": args.profile}

    values = PROFILES[profile]
    payload = {
        "resolved_profile": profile,
        "hardware": info,
        "defaults": values,
        "preserved_env": {key: os.environ[key] for key in values if key in os.environ},
    }
    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(shell_defaults(profile, values, info))


if __name__ == "__main__":
    main()
