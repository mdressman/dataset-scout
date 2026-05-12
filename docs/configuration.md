# Configuration

`dataset-scout` keeps configuration explicit. There is no global state.
The CLI builds a `ScoutContext` from environment variables (and a
`.env` file in the current working directory); library callers
construct one directly.

```python
from dataset_scout import ScoutContext, recon

ctx = ScoutContext.from_env()           # CLI-style: reads env + .env
result = recon("find prompt injection corpora", ctx=ctx)
```

---

## 1. LLM provider

`dataset-scout` is provider-agnostic: the LLM call sites dispatch
through [LiteLLM](https://docs.litellm.ai/) by **model-id prefix**.
Set `DATASET_SCOUT_MODEL` to any litellm-style id and the matching
auth path is used. No code change, no rebuild.

| `DATASET_SCOUT_MODEL=` | Auth | When to pick it |
|---|---|---|
| `github_copilot/<model>` | OAuth device-code (one-time, browser flow) | **Recommended for individuals.** Reuses your existing GitHub Copilot subscription — no separate API key, no Azure account. *Caveat below.* |
| `github/<model>` | `GITHUB_TOKEN` env var (PAT with `models:read`) | **Recommended free path.** Free GitHub Models tier — tighter rate limits than Copilot, but no subscription required. |
| `openai/<model>` | `OPENAI_API_KEY` | You already have an OpenAI account. |
| `anthropic/<model>` | `ANTHROPIC_API_KEY` | You already have an Anthropic account. |
| `azure/<deployment>` | Microsoft Entra (`az login`) — see § 1c | Enterprise / Azure-hosted. **Original auth path; still fully supported.** Equivalent to setting the `AZURE_OPENAI_*` vars in § 1c. |

If `DATASET_SCOUT_MODEL` is unset, the legacy `AZURE_OPENAI_ENDPOINT`
+ `AZURE_OPENAI_DEPLOYMENT` block (§ 1c) is consulted as a fallback.
If neither is configured the pipeline runs in **metadata-only mode**
— see [concepts.md](concepts.md#2-modes-full-vs-metadata-only).

`recon`, `decompose`, and `inspect` accept `--model` to override
`DATASET_SCOUT_MODEL` per-invocation.

### 1a. GitHub Copilot (no extra signup)

```bash
cp .env.example .env
# Then edit .env:
#   DATASET_SCOUT_MODEL=github_copilot/gpt-5-mini
```

First call triggers an OAuth device-code flow: litellm prints a code
and a URL (`github.com/login/device`), you paste the code, and the
token is cached at `~/.config/litellm/github_copilot/` (override with
`GITHUB_COPILOT_TOKEN_DIR`). Subsequent calls refresh silently.

> ⚠️ **Terms-of-service caveat.** GitHub's Copilot terms scope it to
> *"code suggestions in your editor and similar features."* Using the
> Copilot endpoint for batch agentic workflows like dataset recon is
> a gray area; you assume the policy risk. The free GitHub Models
> tier (§ 1b) has no such caveat.

### 1b. GitHub Models (free tier)

```bash
cp .env.example .env
# Then edit .env:
#   DATASET_SCOUT_MODEL=github/gpt-4o-mini
#   GITHUB_TOKEN=github_pat_...
```

Create a fine-grained PAT at
<https://github.com/settings/personal-access-tokens> with the
`models:read` permission. Free tier; rate limits are tight enough
that recon (which fans out the strategy assessor) may need a couple
of retries on busy days.

### 1c. Azure OpenAI (Entra auth)

The original auth path; still fully supported. Use this when your
team already has an AOAI deployment or you need enterprise-grade
isolation.

#### Local development

```bash
az login                                                        # one-time
cp .env.example .env
# Then edit .env:
#   AZURE_OPENAI_ENDPOINT=https://your-aoai-resource.openai.azure.com/
#   AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
```

The `DefaultAzureCredential` chain checks (in order): env-var
service-principal, workload identity, managed identity,
shared-token-cache, Azure CLI (`az login`), and interactive browser.
Whichever wins first is used. Tokens are cached internally — you
don't need to re-`az login` between runs unless your token expires.

#### CI / Azure-hosted

Either:

- **Service principal** — set `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`,
  and `AZURE_CLIENT_SECRET` (the credential picks them up
  automatically), **or**
- **Managed identity** — no env vars needed; the credential
  discovers it.

#### Recognised env vars

| Variable | Purpose | Default |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | e.g. `https://my-aoai.openai.azure.com` (trailing slash trimmed) | — |
| `AZURE_OPENAI_DEPLOYMENT` | Deployment name (NOT model name; e.g. `gpt-4o-mini`) | — |
| `AZURE_OPENAI_API_VERSION` | Pin a specific API version | `2024-10-21` |

When both `DATASET_SCOUT_MODEL` is unset **and** either
`AZURE_OPENAI_ENDPOINT` or `_DEPLOYMENT` is missing, the pipeline
runs in **metadata-only mode** — see
[concepts.md](concepts.md#2-modes-full-vs-metadata-only).

### 1d. Embeddings (label-intent fit)

The embedding-fit stage has its own backend selector, independent of
the chat LLM:

| Variable | Values | Default |
|---|---|---|
| `DATASET_SCOUT_EMBEDDING_BACKEND` | `sbert` (local), `aoai`, `none` | `sbert` |
| `DATASET_SCOUT_EMBEDDING_MODEL` | HF repo id (sbert) or AOAI deployment name | sbert: `sentence-transformers/all-MiniLM-L6-v2` |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | AOAI deployment name (consulted only when backend=aoai) | — |

`sbert` is the default because it's local and works for users with
no Azure account. Install the optional extra to enable it:
`uv tool install 'dataset-scout[local-embeddings] @ git+https://github.com/mdressman/dataset-scout'`
(or `pip install dataset-scout[local-embeddings]`). When
sentence-transformers isn't installed the stage no-ops cleanly and
the rest of the pipeline still runs.

Mixed-provider setups work — e.g. chat via `github_copilot/...` +
embeddings via `aoai` (just set `AZURE_OPENAI_ENDPOINT` +
`AZURE_OPENAI_EMBEDDING_DEPLOYMENT`).

---

## 2. HuggingFace

Public datasets are accessible without a token, but the unauthenticated
rate limits are aggressive enough to hurt during real recon. Setting
a token is recommended.

```bash
# .env
HUGGINGFACE_HUB_TOKEN=hf_...
```

Either `HUGGINGFACE_HUB_TOKEN` or `HF_TOKEN` works.

Get a token at <https://huggingface.co/settings/tokens>. Read-only
permissions are sufficient.

---

## 3. Kaggle

Set `KAGGLE_USERNAME` and `KAGGLE_KEY` to enable Kaggle search.
Kaggle source is discovery-only — `stream_sample`/`stream_rows` are
not supported; components from Kaggle materialise as
`unsupported_source` in `curate` with an actionable hint.

---

## 4. Paths

| Variable | What it overrides | Default (Linux/macOS) | Default (Windows) |
|---|---|---|---|
| `DATASET_SCOUT_CACHE_DIR` | SQLite cache and the judge disk cache | `~/.cache/dataset-scout/` | `%LOCALAPPDATA%\dataset-scout\cache\` |
| `DATASET_SCOUT_CONFIG_DIR` | Config TOML | `~/.config/dataset-scout/` | `%APPDATA%\dataset-scout\` |
| `DATASET_SCOUT_OUT_DIR` | Default `--out` for `recon` | `./datascout-out/` | `.\datascout-out\` |
| `GITHUB_COPILOT_TOKEN_DIR` | Where litellm caches the Copilot OAuth token | `~/.config/litellm/github_copilot/` | `%APPDATA%\litellm\github_copilot\` |

XDG variables (`XDG_CACHE_HOME`, `XDG_CONFIG_HOME`) are honored on
Unix when the explicit overrides aren't set.

The M10 **judge disk cache** lives under
`<DATASET_SCOUT_CACHE_DIR>/judge/<sha256>.json`, keyed by
`sha256(prompt + axis + model + template_version)`. Cache hits skip
the LLM call entirely and same-row + same-judge_version → byte-identical
verdict. The cache is shared across runs and recipes within the same
workspace. Bumping scout's internal `template_version` invalidates
the cache deliberately — that's a scout-internal concern and is
**not** coordinated with any other tool. See
[`judged-corpus-shape.md`](judged-corpus-shape.md) for the public
record-level surface.

The provider id (resolved via `effective_model_id`) is part of the
cache key for every LLM-backed stage — switching providers does not
serve stale results.

---

## 5. The `.env` auto-load

The CLI loads `.env` from the **current working directory** at
startup using `python-dotenv` with `override=False`. That means:

- Variables already set in your shell **win**. `.env` only fills gaps.
- Drop `.env` in a project root and `cd` there — recon picks it up.
- Library callers do **not** auto-load `.env`; pass an explicit
  `ScoutContext` (use `ScoutContext.from_env(env={"DATASET_SCOUT_MODEL": ..., ...})`
  if you want to control it from a test).

`.env.example` in the repo root documents every recognised variable
with a short comment about when each is needed.

---

## 6. Source enablement

Sources can be disabled even if their auth is configured:

```python
from dataset_scout import ScoutContext
from dataset_scout.context import SourceConfig

ctx = ScoutContext(
    sources={
        "huggingface": SourceConfig(enabled=True),
        "kaggle":      SourceConfig(enabled=False),
        "pwc":         SourceConfig(enabled=False),
    },
)
```

The CLI respects the same map; future `datascout sources enable/disable`
will write to a TOML config file.

---

## 7. Programmatic ScoutContext

```python
from dataset_scout import ScoutContext

# Universal-provider construction (recommended)
ctx = ScoutContext(
    model="github_copilot/gpt-5-mini",
    embedding_backend="sbert",
    api_keys={"HUGGINGFACE_HUB_TOKEN": "hf_..."},
    out_dir=Path("/tmp/scout"),
)

# Legacy AOAI construction (still supported)
ctx = ScoutContext(
    aoai_endpoint="https://my-aoai.openai.azure.com",
    aoai_deployment="gpt-4o-mini",
    aoai_api_version="2024-10-21",
    api_keys={"HUGGINGFACE_HUB_TOKEN": "hf_..."},
    out_dir=Path("/tmp/scout"),
)
```

`ScoutContext` is a frozen Pydantic v2 model — `extra="forbid"`,
mutation raises. The intent is that contexts flow through call sites
unchanged; build a new one if you need different config.
