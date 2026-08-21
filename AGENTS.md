# AGENTS.md

- Use `uv` for all Python interactions: install/sync dependencies with `uv sync --locked --all-groups` and run Python tools via `uv run ...` (do not call `python`, `pip`, `pytest`, etc. directly).
- Validate changes with the same checks as CI (`.github/workflows/ci.yml`):
  - `uv run pyright`
  - `uv run ruff check`
  - `uv run pytest -m "not live"`
- Keep changes focused and avoid running live tests unless explicitly requested.
