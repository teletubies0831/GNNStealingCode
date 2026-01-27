"""Run end-to-end GNNStealing + GNNFingers pipeline from a JSON config."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
import subprocess
from typing import Any, Dict, List


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run GNNStealing + GNNFingers pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser.parse_args()


def _normalize_flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def _build_args(args_dict: Dict[str, Any]) -> List[str]:
    cli_args: List[str] = []
    for key, value in args_dict.items():
        if value is None:
            continue
        flag = _normalize_flag(key)
        if isinstance(value, bool):
            if value:
                cli_args.append(flag)
            continue
        if isinstance(value, list):
            cli_args.extend([flag, ",".join(str(v) for v in value)])
            continue
        cli_args.extend([flag, str(value)])
    return cli_args


def _run_step(command: List[str], cwd: Path, dry_run: bool) -> None:
    print("[pipeline]", " ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=str(cwd), check=True)


def _write_csv(path: Path, rows: List[List[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(str(cell) for cell in row) for row in rows]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    repo_root = Path(__file__).resolve().parents[1]
    code_dir = repo_root

    if "suspect_list" in config:
        csv_path = Path(config["suspect_list"]["path"])
        _write_csv(csv_path, config["suspect_list"]["rows"])

    if "source_list" in config:
        csv_path = Path(config["source_list"]["path"])
        _write_csv(csv_path, config["source_list"]["rows"])

    steps = config.get("steps", [])
    for step in steps:
        name = step["name"]
        script = step["script"]
        args_dict = step.get("args", {})
        cmd = ["python", script] + _build_args(args_dict)
        print(f"[pipeline] Running step: {name}")
        _run_step(cmd, cwd=code_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
