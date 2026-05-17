# Contributing to Post_MD

Thanks for your interest. Post_MD is an open-source project under dual
licensing (AGPL-3.0 + commercial), and a few rules follow from that.

## Contributor License Agreement (CLA)

Because Post_MD is dual-licensed, every contributor must agree that their
contributions may be relicensed by the maintainers under the commercial
license alongside the AGPL-3.0 grant. **By submitting a pull request, you
certify the following:**

1. You are the author of the contribution, OR you have the right to submit
   it under both the AGPL-3.0 and the Post_MD commercial license.
2. You grant the Post_MD maintainers a perpetual, worldwide, royalty-free
   license to relicense your contribution under the commercial license in
   addition to the AGPL-3.0.
3. Your contribution does not include code under a license incompatible with
   the AGPL-3.0 (e.g., GPL-only code without a compatible exception) or with
   a commercial relicense (e.g., other strong-copyleft code).

A formal CLA document will be added before the first public release; until
then, opening a pull request constitutes acceptance of the terms above.

## Development setup

```bash
pip install -e ".[dev]"
pre-commit install
pytest -q
```

## Style

- `ruff` for lint + format. Run `ruff check . && ruff format .` before pushing.
- `mypy src/post_md` for type-checking. Strict mode is off for now but new
  code should be annotated where it doesn't add noise.
- Tests live under `tests/`. Aim for fast unit tests; integration tests that
  need real trajectory files belong under `tests/data/` with checked-in
  fixtures small enough to ship (≤ 50 KB each).

## Scope guidelines

- New trajectory or topology formats are welcome — implement against the
  public spec rather than wrapping an existing library.
- Performance-critical inner loops may move to Cython or Rust extensions,
  but the pure-Python implementation must remain as a fallback.
- The selection grammar is intentionally small in v1; geometric selections
  (`around 5.0 X`) are deferred to v2.
