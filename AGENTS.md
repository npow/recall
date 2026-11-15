# AGENTS.md

- Shadow Canvas work lives on branch `shadow-canvas`: proposal `proposals/shadow-canvas.md`; code `worldsync/{poses,canvas,lingbot,upstream}.py`; Modal entry points `scripts/modal_shadow_canvas.py`.
- The team's stock LingBot-World2 14B reproduction (protocol, gates, results) is described in `reproduction/README.md`; reuse its verified modules (`lingbot_native`, `lingbot_vae`, `matrix_attention`) rather than duplicating them.
- Pure-Python tests (no GPU): `uv run -q --no-project --with pytest --with numpy python -m pytest -q tests/test_poses.py tests/test_canvas.py`. Most other tests need torch.
- Lint new code: `uv run -q --no-project --with ruff ruff check worldsync scripts tests --select F,E9`.
- Modal (team workspace profile `nissan-pow`): `modal run --profile nissan-pow scripts/modal_shadow_canvas.py::validate_streaming` (also `::smoke_turn`, `::download_weights`). Weights volume `worldsync-lingbot-weights` (14B at `/weights/14b-causal-fast`); results volume `worldsync-lingbot-results`.
- LingBot source is pinned to `1895d300d8ac936401689b26389f51cbd36530eb` and 14B weights to `5c33dd40b213598c418fd25bff30fdbd23fd38a7`.
- Commit messages end with the Devin "Generated with" line and Co-Authored-By trailer (see `git log`). Ask before pushing to `main`.
