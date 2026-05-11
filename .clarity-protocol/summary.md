# dataset-scout

A CLI tool that automates dataset discovery and reframing for ML practitioners. You write a brief describing the dataset you need, and it runs an 8-stage pipeline: brief parsing, LLM decomposition into search directions, multi-source search (HuggingFace, Kaggle, Semantic Scholar/arXiv), cheap metadata probes, embedding-based shortlisting, per-candidate strategy assessment with real sample rows, coverage gap analysis, and self-contained HTML+Markdown report generation with a draft recipe.

The core value is turning a manual, hours-long search-and-reframe loop into a 2-minute automated pipeline that produces auditable, shippable output. Key design principles: no aggregate quality scores (per-signal evidence instead), proxies are honest by default via `label_kind`, receipts tie every claim to actual column names and sample rows, and coverage gaps are first-class deliverables.

Sweet spot is AI-security detection work where direct-fit datasets rarely exist and creative reframings of adjacent data are the norm. Experimental downstream paths include corpus materialization (`curate` with MinHash dedup, leakage-aware splits, lockfile audit trail) and LLM-as-judge label rescue.

v0.0.1, Pre-Alpha, MIT licensed, Python 3.11+. Azure OpenAI via Entra auth, HuggingFace Hub API, source plugins via entry-point group.