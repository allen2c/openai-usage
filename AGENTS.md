# AGENTS.md

One `Usage` model that normalizes usage from every OpenAI API, plus google-genai
and litellm, and prices it against live model catalogs.

## Commands

```bash
make install   # poetry install --all-extras --all-groups
make fmt       # isort + black + ruff --fix
make check     # lint + pyright + pytest
```

Run `make check` before committing. It must be clean.

## Layout

| Path | What it is |
| --- | --- |
| `openai_usage/__init__.py` | `Usage`, `Usage.from_openai`, cost estimation |
| `openai_usage/pricing/` | Catalog fetching, caching, and rate lookup |
| `openai_usage/extra/` | Deprecated import paths, kept as shims |
| `openai_usage/models.json` | Bundled OpenRouter snapshot, used offline |
| `docs/` | The real documentation |

## Rules

- **Compatibility first.** This is a published PyPI package. Keep old import
  paths working via shims, give new model fields defaults so stored JSON still
  validates, and keep existing keyword arguments. Never ship a fatal
  incompatibility.
- **No comments in `pyproject.toml`.** Put the rationale in the commit message.
- **File layout:** docstring, imports, constants, public functions, public
  classes, private functions and classes.
- **`google-genai` and `litellm` are extras.** Never import them at module
  scope; detect them through `sys.modules` and quote their type annotations.
- Line length is 88 for both `black` and `ruff`.

## Docs

Start at [`docs/index.md`](docs/index.md).

| Topic | Page |
| --- | --- |
| Install, extras | [`docs/installation.md`](docs/installation.md) |
| Collecting and aggregating usage | [`docs/tracking-usage.md`](docs/tracking-usage.md) |
| Every accepted usage type | [`docs/supported-usage.md`](docs/supported-usage.md) |
| Pricing, catalogs, caching | [`docs/estimating-costs.md`](docs/estimating-costs.md) |
| Full API surface | [`docs/api-reference.md`](docs/api-reference.md) |
