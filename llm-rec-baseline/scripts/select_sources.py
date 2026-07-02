#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import time
import urllib.request


PYPI_SOURCES = [
    ("official", "https://pypi.org/simple"),
    ("tsinghua", "https://pypi.tuna.tsinghua.edu.cn/simple"),
    ("ustc", "https://mirrors.ustc.edu.cn/pypi/simple"),
    ("aliyun", "https://mirrors.aliyun.com/pypi/simple"),
    ("tencent", "https://mirrors.cloud.tencent.com/pypi/simple"),
    ("huawei", "https://repo.huaweicloud.com/repository/pypi/simple"),
]

HF_SOURCES = [
    ("official", "https://huggingface.co"),
    ("hf-mirror", "https://hf-mirror.com"),
]


def probe(url: str, timeout: float) -> tuple[bool, float, str]:
    start = time.monotonic()
    try:
        req = urllib.request.Request(url, method="GET", headers={"User-Agent": "llm-rec-baseline-source-probe"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ok = 200 <= resp.status < 400
            return ok, time.monotonic() - start, f"HTTP {resp.status}"
    except Exception as exc:
        return False, time.monotonic() - start, exc.__class__.__name__


def choose(kind: str, timeout: float, preferred: str | None) -> dict:
    if preferred:
        sources = [("env", preferred)]
    elif kind == "pypi":
        sources = PYPI_SOURCES
    elif kind == "hf":
        sources = HF_SOURCES
    else:
        raise ValueError(kind)

    results = []
    for name, url in sources:
        check_url = url.rstrip("/") + ("/pip/" if kind == "pypi" else "/api/models/OpenOneRec/OneReason-0.8B-pretrain-competition")
        ok, elapsed, detail = probe(check_url, timeout)
        result = {"name": name, "url": url, "ok": ok, "seconds": round(elapsed, 3), "detail": detail}
        results.append(result)
        if ok:
            return {"kind": kind, "selected": result, "results": results}

    return {"kind": kind, "selected": None, "results": results}


def shell_quote_assignment(key: str, value: str) -> str:
    return f"export {key}={shlex.quote(value)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("SOURCE_PROBE_TIMEOUT", "4")))
    parser.add_argument("--format", choices=["json", "shell"], default="json")
    args = parser.parse_args()

    pypi = choose("pypi", args.timeout, os.environ.get("PIP_INDEX_URL") or os.environ.get("UV_INDEX_URL"))
    hf = choose("hf", args.timeout, os.environ.get("HF_ENDPOINT"))
    payload = {"pypi": pypi, "huggingface": hf}

    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    lines = []
    if pypi["selected"]:
        index_url = pypi["selected"]["url"]
        lines.append(shell_quote_assignment("PIP_INDEX_URL", index_url))
        lines.append(shell_quote_assignment("UV_INDEX_URL", index_url))
        lines.append(shell_quote_assignment("SELECTED_PYPI_SOURCE", pypi["selected"]["name"]))
    if hf["selected"]:
        lines.append(shell_quote_assignment("HF_ENDPOINT", hf["selected"]["url"]))
        lines.append(shell_quote_assignment("SELECTED_HF_SOURCE", hf["selected"]["name"]))
    lines.append(shell_quote_assignment("SOURCE_PROBE_REPORT", json.dumps(payload, ensure_ascii=False, sort_keys=True)))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
