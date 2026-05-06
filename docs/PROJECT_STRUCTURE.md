# Project Structure

This repository is organized around a small set of durable top-level areas:

1. `predictdesign/`
   Core library code.

2. `scripts/`
   Operational entry points:
   - `scripts/benchmark/`
   - `scripts/training/`
   - `scripts/ops/`

3. `examples/`
   Small runnable examples.

4. `tests/`
   Automated verification.

5. `data/`
   Local dataset-like inputs such as `data/acg_nap/`.

6. `vendor/`
   Vendored third-party benchmark code such as `vendor/prefetch-kv-mas/`.

7. `results/`
   Generated experiment outputs and archived smoke runs.

8. `docs/`
   Lightweight maintenance notes.

## Maintenance Rules

- Keep generated artifacts out of `predictdesign/`, `tests/`, and `examples/`.
- Place new scripts by intent:
  - benchmark runner -> `scripts/benchmark/`
  - model training/eval -> `scripts/training/`
  - cleanup, monitoring, shell orchestration -> `scripts/ops/`
- Prefer path constants from `predictdesign.paths` instead of repeating literal repository paths.

## Cleanup

Dry run:

```bash
python scripts/cleanup_workspace.py
```

Apply cleanup:

```bash
python scripts/cleanup_workspace.py --execute
```

Archive old smoke runs too:

```bash
python scripts/cleanup_workspace.py --execute --archive-smoke-results
```
