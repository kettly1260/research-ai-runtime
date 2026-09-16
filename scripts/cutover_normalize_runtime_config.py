#!/usr/bin/env python3
"""Normalise the OVMS runtime model config for the legacy -> modular cutover.

Background
----------
The legacy gateway writes **raw** model names (``bge-m3-i8``) into the OVMS
runtime config. The modular gateway's DeviceBroker writes **hardware-suffixed
aliases** (``bge-m3-i8__gpu`` / ``bge-m3-i8__cpu``) and only ever *appends* when
the alias is absent. Starting the modular gateway against an un-normalised
legacy config therefore produces two OVMS models over the same ``base_path``:

    bge-m3-i8        base_path=/models/bge-m3-i8
    bge-m3-i8__cpu   base_path=/models/bge-m3-i8     <-- duplicate residency

Worse, the broker's in-memory ``model_loaded`` map starts empty, so the
raw-name entry is invisible to the idle sweeper and to ``MAX_LOADED_MODELS``
accounting -- it can never be evicted.

This script performs the safe transition: atomically replace the runtime config
with an empty ``model_config_list`` so OVMS unloads everything, then let the
broker add exactly the aliases it tracks, on demand. Pre-writing the aliases is
deliberately NOT done, because those entries would be untracked by the broker
and could never be evicted.

No ``POST /v1/config/reload`` is ever issued: OVMS polls the config file
(``file_system_poll_wait_seconds`` defaults to 1s).

Usage
-----
    # read-only: show what would change and verify invariants
    python3 cutover_normalize_runtime_config.py plan  --config /ovms-config/config.json
    python3 cutover_normalize_runtime_config.py check --config /ovms-config/config.json \\
                                                      --ovms-base http://ovms-server:8001

    # perform the transition (writes the file, waits for zero-resident)
    python3 cutover_normalize_runtime_config.py apply --config /ovms-config/config.json \\
                                                      --ovms-base http://ovms-server:8001

Exit codes: 0 = ok, 1 = invariant violation / failure, 2 = usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

DEFAULT_CONFIG = "/ovms-config/config.json"
DEFAULT_OVMS_BASE = "http://127.0.0.1:28331"
ZERO_RESIDENT_PAYLOAD: Dict[str, Any] = {"model_config_list": []}


# --------------------------------------------------------------------------- io

def read_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid OVMS config payload at {path}")
    if not isinstance(payload.get("model_config_list"), list):
        raise ValueError(f"model_config_list missing or not a list at {path}")
    return payload


def write_config_atomic(path: str, payload: Dict[str, Any]) -> None:
    """Same atomicity contract the gateway itself uses (tmp + fsync + replace)."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.cutover.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def fetch_residency(ovms_base: str, timeout: float = 20.0) -> Dict[str, str]:
    """Returns {model_name: state} from the OVMS runtime config endpoint."""
    url = f"{ovms_base.rstrip('/')}/v1/config"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError, OSError) as exc:
        raise RuntimeError(f"cannot read OVMS runtime config from {url}: {exc}") from exc

    residency: Dict[str, str] = {}
    for name, body in payload.items():
        statuses = body.get("model_version_status") if isinstance(body, dict) else None
        state = "UNKNOWN"
        if isinstance(statuses, list) and statuses:
            state = str(statuses[0].get("state", "UNKNOWN")).upper()
        residency[name] = state
    return residency


# ------------------------------------------------------------------- invariants

def entry_names(payload: Dict[str, Any]) -> List[str]:
    names = []
    for item in payload.get("model_config_list", []):
        cfg = item.get("config", {}) if isinstance(item, dict) else {}
        names.append(str(cfg.get("name", "")))
    return names


def base_paths_by_name(payload: Dict[str, Any]) -> Dict[str, str]:
    mapping = {}
    for item in payload.get("model_config_list", []):
        cfg = item.get("config", {}) if isinstance(item, dict) else {}
        name = str(cfg.get("name", ""))
        mapping[name] = str(cfg.get("base_path", ""))
    return mapping


def find_violations(
    payload: Dict[str, Any], residency: Optional[Dict[str, str]] = None
) -> List[str]:
    """Returns human-readable invariant violations (empty list == healthy)."""
    problems: List[str] = []
    names = entry_names(payload)
    paths = base_paths_by_name(payload)

    # 1. No two configured entries may share a base_path (duplicate residency).
    seen: Dict[str, List[str]] = {}
    for name, path in paths.items():
        if path:
            seen.setdefault(path, []).append(name)
    for path, owners in sorted(seen.items()):
        if len(owners) > 1:
            problems.append(
                f"duplicate base_path {path} is declared by {sorted(owners)} "
                "(both would be resident simultaneously)"
            )

    if residency is None:
        return problems

    available = sorted(name for name, state in residency.items() if state == "AVAILABLE")

    # 2. After cutover every resident model must be an alias the broker tracks.
    for name in available:
        if "__" not in name:
            problems.append(
                f"resident model {name!r} uses a raw (legacy) name; the modular broker "
                "only tracks <name>__gpu / <name>__cpu aliases, so this entry would "
                "never be evicted"
            )

    # 3. Every resident model must be declared in the runtime config file.
    for name in available:
        if name not in names:
            problems.append(
                f"resident model {name!r} is not present in the runtime config file"
            )

    # 4. No base_path may be resident more than once.
    by_path: Dict[str, List[str]] = {}
    for name in available:
        path = paths.get(name)
        if path:
            by_path.setdefault(path, []).append(name)
    for path, owners in sorted(by_path.items()):
        if len(owners) > 1:
            problems.append(
                f"base_path {path} is resident {len(owners)} times: {sorted(owners)}"
            )

    return problems


# ----------------------------------------------------------------------- modes

def cmd_plan(args: argparse.Namespace) -> int:
    payload = read_config(args.config)
    names = entry_names(payload)
    print(f"config            : {args.config}")
    print(f"configured models : {len(names)}")
    for name in names:
        print(f"  - {name}")
    raw_names = [n for n in names if n and "__" not in n]
    alias_names = [n for n in names if "__" in n]
    print(f"raw (legacy) names: {raw_names}")
    print(f"alias names       : {alias_names}")
    print()
    print("target payload    : " + json.dumps(ZERO_RESIDENT_PAYLOAD))
    print(
        "reason            : an empty list makes OVMS unload everything, so no raw name "
        "can coexist with the aliases the broker adds on demand"
    )
    problems = find_violations(payload)
    if problems:
        print()
        print("current violations:")
        for problem in problems:
            print(f"  ! {problem}")
        return 1
    print()
    print("current config declares no duplicate base_path: OK")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    payload = read_config(args.config)
    residency = fetch_residency(args.ovms_base, timeout=args.http_timeout)
    print(f"config    : {args.config}")
    print(f"ovms      : {args.ovms_base}")
    print("residency :")
    for name, state in sorted(residency.items()):
        print(f"  - {name:45s} {state}")
    problems = find_violations(payload, residency)
    print()
    if problems:
        print("INVARIANT VIOLATIONS:")
        for problem in problems:
            print(f"  ! {problem}")
        return 1
    print("alias transition invariants: OK")
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    payload = read_config(args.config)
    before_names = entry_names(payload)
    before_residency = fetch_residency(args.ovms_base, timeout=args.http_timeout)
    before_available = sorted(n for n, s in before_residency.items() if s == "AVAILABLE")

    print("== BEFORE ==")
    print(f"config entries   : {before_names}")
    print(f"ovms available   : {before_available}")

    # Refuse to run if the config is already alias-only AND healthy: nothing to do.
    if not before_names:
        print()
        print("config is already zero-resident; nothing to normalise")
        problems = find_violations(payload, before_residency)
        return 0 if not problems else 1

    backup = f"{args.config}.bak-cutover-normalize-{time.strftime('%Y%m%d-%H%M%S')}"
    if args.apply:
        with open(args.config, "rb") as src, open(backup, "wb") as dst:
            dst.write(src.read())
        print(f"backup written   : {backup}")
        write_config_atomic(args.config, ZERO_RESIDENT_PAYLOAD)
        print(f"config replaced  : {json.dumps(ZERO_RESIDENT_PAYLOAD)}")
    else:
        print("dry-run (no --apply): config left untouched")

    print()
    print("== waiting for zero-resident (OVMS filesystem polling, no reload API) ==")
    deadline = time.time() + args.timeout
    last_available: List[str] = before_available
    while time.time() < deadline:
        residency = fetch_residency(args.ovms_base, timeout=args.http_timeout)
        last_available = sorted(n for n, s in residency.items() if s == "AVAILABLE")
        if not last_available:
            print("zero-resident reached")
            break
        time.sleep(2.0)
    else:
        print(f"TIMEOUT after {args.timeout}s; still AVAILABLE: {last_available}")
        return 1

    print()
    print("== AFTER ==")
    final_payload = read_config(args.config)
    final_residency = fetch_residency(args.ovms_base, timeout=args.http_timeout)
    print(f"config entries   : {entry_names(final_payload)}")
    print(f"ovms available   : {sorted(n for n, s in final_residency.items() if s == 'AVAILABLE')}")

    problems = find_violations(final_payload, final_residency)
    print()
    if problems:
        print("INVARIANT VIOLATIONS:")
        for problem in problems:
            print(f"  ! {problem}")
        print()
        print(f"rollback: cp -p {backup} {args.config}")
        return 1
    print("normalisation complete; invariants hold")
    print()
    print("next: start the modular gateway. It will add only <name>__gpu / <name>__cpu")
    print("      entries on demand. First request per model pays a cold load.")
    return 0


# ------------------------------------------------------------------------ main

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["plan", "check", "apply"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--ovms-base", default=DEFAULT_OVMS_BASE)
    parser.add_argument("--timeout", type=float, default=180.0, help="zero-resident wait budget (s)")
    parser.add_argument("--http-timeout", type=float, default=20.0)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="for mode=apply: actually write the config (omit for a dry run)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "plan":
            return cmd_plan(args)
        if args.mode == "check":
            return cmd_check(args)
        return cmd_apply(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
