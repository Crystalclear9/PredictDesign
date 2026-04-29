from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from predictdesign.benchmark.multiagentbench import MultiAgentBenchAdapter
from predictdesign.benchmark.rich_log import train_mlp_on_rich_log, write_rich_log
from predictdesign.config import LLMApiConfig
from run_marble_hitk_benchmark import (
    DEFAULT_TASK_IDS,
    _append_log,
    _base_env,
    _parse_task_ids,
    _python_bin,
    _run_coding,
    _run_research,
    _run_werewolf,
    _timestamp,
)


def _default_results_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "results" / "parallel_api_rich_logs" / timestamp


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _console(message: str) -> None:
    print(message, flush=True)


def _status(run_log_path: Path, message: str) -> None:
    _console(message)
    _append_log(run_log_path, f"[{_timestamp()}] {message}")


def _build_api_config(args: argparse.Namespace, prefix: str) -> LLMApiConfig:
    defaults = LLMApiConfig()
    env_prefix = prefix.upper()
    api_key = (
        getattr(args, f"{prefix}_api_key")
        or args.api_key
        or _env_first(f"{env_prefix}_API_KEY", f"{env_prefix}_LLM_API_KEY")
        or _env_first("OPENAI_API_KEY", "LLM_API_KEY", "PREDICTDESIGN_LLM_API_KEY")
        or defaults.api_key
    )
    base_url = (
        getattr(args, f"{prefix}_base_url")
        or args.base_url
        or _env_first(f"{env_prefix}_BASE_URL", f"{env_prefix}_API_BASE", f"{env_prefix}_LLM_BASE_URL")
        or _env_first("OPENAI_BASE_URL", "OPENAI_API_BASE", "LLM_BASE_URL", "PREDICTDESIGN_LLM_BASE_URL")
        or defaults.base_url
    )
    model = (
        getattr(args, f"{prefix}_model")
        or args.model
        or _env_first(f"{env_prefix}_MODEL", f"{env_prefix}_LLM_MODEL")
        or _env_first("LLM_MODEL", "PREDICTDESIGN_LLM_MODEL")
        or defaults.model
    )
    return LLMApiConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
    )


def _safe_api_summary(config: LLMApiConfig) -> dict[str, object]:
    return {
        "api_key": "***" if config.api_key else "",
        "base_url": config.base_url,
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout": config.timeout,
        "max_retries": config.max_retries,
    }


def _load_episodes(
    *,
    coding_outputs: list[Path],
    research_outputs: list[Path],
    werewolf_outputs: list[Path],
    context_dim: int,
    hidden_dim: int,
    device: str,
):
    adapter = MultiAgentBenchAdapter(context_dim=context_dim, hidden_dim=hidden_dim, device=device)
    episodes = []
    for path in coding_outputs:
        episodes.extend(adapter.load_coding_from_output_jsonl(path))
    for path in research_outputs:
        episodes.extend(adapter.load_research_from_output_jsonl(path))
    for path in werewolf_outputs:
        episodes.extend(adapter.load_werewolf_from_checkpoints(path))
    return episodes


def _write_original_manifest(results_dir: Path, outputs: dict[str, list[Path]]) -> Path:
    manifest_path = results_dir / "original_logs_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "coding_output_jsonl": [str(path) for path in outputs.get("coding", [])],
                "research_output_jsonl": [str(path) for path in outputs.get("research", [])],
                "werewolf_checkpoint_dirs": [str(path) for path in outputs.get("werewolf", [])],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest_path


def _check_runtime_python(python_bin: str) -> None:
    probe = (
        "import sys; "
        "import fire; "
        "import ruamel.yaml; "
        "print(sys.executable)"
    )
    try:
        subprocess.run(
            [python_bin, "-c", probe],
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(
            "The selected Python cannot run MARBLE dependencies. "
            f"python_bin={python_bin}. "
            "Install missing packages into that interpreter, or pass "
            "--python-bin C:\\Users\\70454\\Desktop\\PredictDesign\\.venv\\Scripts\\python.exe. "
            f"Details: {details}"
        ) from exc


def _scenario_env(config: LLMApiConfig, args: argparse.Namespace) -> dict[str, str]:
    env = _base_env(config)
    env["PREDICTDESIGN_TEE_SUBPROCESS"] = "1" if args.tee_subprocess_output else "0"
    return env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run coding/research/werewolf concurrently with separate runtime APIs and export one rich log."
    )
    parser.add_argument("--coding-task-ids", nargs="*", type=int, default=list(DEFAULT_TASK_IDS))
    parser.add_argument("--research-task-ids", nargs="*", type=int, default=list(DEFAULT_TASK_IDS))
    parser.add_argument("--werewolf-games", type=int, default=10)
    parser.add_argument("--skip-coding", action="store_true")
    parser.add_argument("--skip-research", action="store_true")
    parser.add_argument("--skip-werewolf", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--results-dir", type=str, default=str(_default_results_dir()))
    parser.add_argument(
        "--python-bin",
        type=str,
        default=None,
        help="Python used for MARBLE subprocesses. Defaults to the project .venv when it exists.",
    )
    parser.add_argument("--output-log", type=str, default=None)
    parser.add_argument(
        "--log-version",
        choices=("rich", "original", "both"),
        default="rich",
        help="rich saves the detailed normalized JSONL; original saves a manifest pointing to raw scenario outputs.",
    )
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument(
        "--tee-subprocess-output",
        action="store_true",
        help="Also stream each MARBLE subprocess log line to the terminal. Logs are always saved to files.",
    )
    parser.add_argument("--context-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--coding-api-key", type=str, default=None)
    parser.add_argument("--coding-base-url", type=str, default=None)
    parser.add_argument("--coding-model", type=str, default=None)
    parser.add_argument("--research-api-key", type=str, default=None)
    parser.add_argument("--research-base-url", type=str, default=None)
    parser.add_argument("--research-model", type=str, default=None)
    parser.add_argument("--werewolf-api-key", type=str, default=None)
    parser.add_argument("--werewolf-base-url", type=str, default=None)
    parser.add_argument("--werewolf-model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=LLMApiConfig().temperature)
    parser.add_argument("--max-tokens", type=int, default=LLMApiConfig().max_tokens)
    parser.add_argument("--timeout", type=float, default=LLMApiConfig().timeout)
    parser.add_argument("--max-retries", type=int, default=LLMApiConfig().max_retries)
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=LLMApiConfig().retry_backoff_seconds,
    )

    parser.add_argument("--train-mlp", action="store_true")
    parser.add_argument("--mlp-output-dir", type=str, default=None)
    parser.add_argument("--mlp-max-samples", type=int, default=0)
    parser.add_argument("--mlp-epochs", type=int, default=60)
    parser.add_argument(
        "--mlp-feature-dim",
        type=int,
        default=384,
        help="Fallback text-vector dimension if sentence-transformers cannot be loaded.",
    )
    parser.add_argument(
        "--mlp-hidden-dim",
        type=int,
        default=0,
        help="Legacy single-width override. Set 0 to use automatic hidden dimensions.",
    )
    parser.add_argument(
        "--mlp-hidden-dims",
        nargs="*",
        type=int,
        default=None,
        help="Optional explicit hidden dimensions, for example: --mlp-hidden-dims 1024 512",
    )
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mlp-device", type=str, default="auto")
    parser.add_argument("--mlp-sentence-transformer-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--mlp-sentence-transformer-batch-size", type=int, default=64)
    parser.add_argument("--mlp-repeat-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--mlp-label-mode",
        choices=("action_type", "action_signature"),
        default="action_type",
    )
    args = parser.parse_args()
    if args.train_mlp and args.log_version == "original":
        raise SystemExit("--train-mlp requires --log-version rich or --log-version both.")
    if args.python_bin:
        os.environ["PREDICTDESIGN_PYTHON_BIN"] = str(Path(args.python_bin).resolve())
    os.environ["PREDICTDESIGN_RESUME"] = "1" if args.resume else "0"
    runtime_python = _python_bin()
    _console(f"[startup] checking runtime python: {runtime_python}")
    _check_runtime_python(runtime_python)

    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = results_dir / "logs" / "parallel_api_run.log"
    _append_log(run_log_path, f"\n===== {_timestamp()} PARALLEL RUN START =====")
    output_log = Path(args.output_log).resolve() if args.output_log else results_dir / "rich_log.jsonl"
    _console(f"[startup] results_dir={results_dir}")
    _console(f"[startup] master_log={run_log_path}")
    _console(f"[startup] rich_log={output_log}")

    configs = {
        "coding": _build_api_config(args, "coding"),
        "research": _build_api_config(args, "research"),
        "werewolf": _build_api_config(args, "werewolf"),
    }
    summary = {
        "results_dir": str(results_dir),
        "python_bin": runtime_python,
        "resume": bool(args.resume),
        "tee_subprocess_output": bool(args.tee_subprocess_output),
        "run_log": str(run_log_path),
        "output_log": str(output_log),
        "apis": {name: _safe_api_summary(config) for name, config in configs.items()},
        "outputs": {},
    }
    _append_log(run_log_path, json.dumps(summary, indent=2, ensure_ascii=False))
    (results_dir / "parallel_api_config_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _console(
        "[startup] tasks "
        f"coding={[] if args.skip_coding else _parse_task_ids(args.coding_task_ids)} "
        f"research={[] if args.skip_research else _parse_task_ids(args.research_task_ids)} "
        f"werewolf_games={0 if args.skip_werewolf else args.werewolf_games} "
        f"resume={args.resume} tee={args.tee_subprocess_output}"
    )

    jobs = {}
    max_workers = max(1, min(args.max_workers, 3))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if not args.skip_coding:
            _status(run_log_path, "[submit] coding")
            jobs[
                executor.submit(
                    _run_coding,
                    _parse_task_ids(args.coding_task_ids),
                    configs["coding"],
                    results_dir,
                    _scenario_env(configs["coding"], args),
                )
            ] = "coding"
        if not args.skip_research:
            _status(run_log_path, "[submit] research")
            jobs[
                executor.submit(
                    _run_research,
                    _parse_task_ids(args.research_task_ids),
                    configs["research"],
                    results_dir,
                    _scenario_env(configs["research"], args),
                )
            ] = "research"
        if not args.skip_werewolf:
            _status(run_log_path, "[submit] werewolf")
            jobs[
                executor.submit(
                    _run_werewolf,
                    args.werewolf_games,
                    configs["werewolf"],
                    results_dir,
                    _scenario_env(configs["werewolf"], args),
                )
            ] = "werewolf"

        if not jobs:
            raise SystemExit("All scenarios were skipped; nothing to run.")

        outputs: dict[str, list[Path]] = {"coding": [], "research": [], "werewolf": []}
        errors: dict[str, str] = {}
        for future in as_completed(jobs):
            scenario = jobs[future]
            try:
                outputs[scenario] = future.result()
            except Exception as exc:  # noqa: BLE001 - keep partial state for resume.
                errors[scenario] = repr(exc)
                event = {
                    "scenario": scenario,
                    "status": "failed",
                    "error": repr(exc),
                }
                _console(json.dumps(event, ensure_ascii=False))
                _append_log(run_log_path, f"[{_timestamp()}] {json.dumps(event, ensure_ascii=False)}")
                continue
            event = {
                "scenario": scenario,
                "status": "completed",
                "output_count": len(outputs[scenario]),
                "outputs": [str(path) for path in outputs[scenario]],
            }
            _console(json.dumps(event, ensure_ascii=False))
            _append_log(run_log_path, f"[{_timestamp()}] {json.dumps(event, ensure_ascii=False)}")

    summary["outputs"] = {name: [str(path) for path in paths] for name, paths in outputs.items()}
    if errors:
        summary["errors"] = errors
        summary_path = results_dir / "parallel_api_run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        _append_log(run_log_path, f"[{_timestamp()}] partial summary written: {summary_path}")
        _append_log(run_log_path, f"===== {_timestamp()} PARALLEL RUN FAILED =====")
        raise SystemExit(
            "One or more scenarios failed. Check logs under "
            f"{results_dir / 'logs'} and rerun with --resume."
        )
    if args.log_version in {"original", "both"}:
        manifest_path = _write_original_manifest(results_dir, outputs)
        summary["original_logs_manifest"] = str(manifest_path)
        _append_log(run_log_path, f"[{_timestamp()}] original manifest written: {manifest_path}")
        _console(json.dumps({"original_logs_manifest": str(manifest_path)}, ensure_ascii=False))

    rich_result = None
    if args.log_version in {"rich", "both"}:
        _status(run_log_path, "[rich-log] loading scenario outputs")
        episodes = _load_episodes(
            coding_outputs=outputs["coding"],
            research_outputs=outputs["research"],
            werewolf_outputs=outputs["werewolf"],
            context_dim=args.context_dim,
            hidden_dim=args.hidden_dim,
            device=args.device,
        )
        _status(run_log_path, f"[rich-log] writing {len(episodes)} episodes -> {output_log}")
        rich_result = write_rich_log(
            output_log,
            episodes,
            context_dim=args.context_dim,
            device=args.device,
        )
        summary["rich_log"] = asdict(rich_result)
        _append_log(run_log_path, f"[{_timestamp()}] rich log written: {rich_result.path}")
        _console(
            f"[rich-log] done path={rich_result.path} "
            f"records={rich_result.record_count} episodes={rich_result.episode_count}"
        )

    if args.train_mlp:
        assert rich_result is not None
        _status(
            run_log_path,
            "[mlp] start "
            f"device={args.mlp_device} samples={args.mlp_max_samples} "
            f"epochs={args.mlp_epochs} feature_dim={args.mlp_feature_dim} hidden_dim={args.mlp_hidden_dim}",
        )
        mlp_result = train_mlp_on_rich_log(
            log_path=rich_result.path,
            output_dir=args.mlp_output_dir or str(results_dir / "rich_log_mlp"),
            max_samples=args.mlp_max_samples,
            feature_dim=args.mlp_feature_dim,
            hidden_dim=args.mlp_hidden_dim,
            epochs=args.mlp_epochs,
            learning_rate=args.mlp_learning_rate,
            seed=args.seed,
            label_mode=args.mlp_label_mode,
            device=args.mlp_device,
            sentence_transformer_model=args.mlp_sentence_transformer_model,
            sentence_transformer_batch_size=args.mlp_sentence_transformer_batch_size,
            hidden_dims=tuple(args.mlp_hidden_dims) if args.mlp_hidden_dims else None,
            repeat_count=args.mlp_repeat_count,
        )
        summary["mlp"] = asdict(mlp_result)
        _append_log(run_log_path, f"[{_timestamp()}] mlp report written: {mlp_result.report_path}")
        _console(f"[mlp] done report={mlp_result.report_path} chart={mlp_result.chart_path}")

    summary_path = results_dir / "parallel_api_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _append_log(run_log_path, f"[{_timestamp()}] summary written: {summary_path}")
    _append_log(run_log_path, f"===== {_timestamp()} PARALLEL RUN END =====")
    _console(str(summary_path))
    _console(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
