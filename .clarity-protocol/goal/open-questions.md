# Open Questions

## OQ1: Curate end-to-end validation
- **Question:** Has anyone trained a model on a scout-curated corpus and confirmed quality against a hand-built reference?
- **Status:** open
- **Strategy:** research
- **Why it matters:** The curate pipeline is the bridge from discovery to usable training data. Without validation, users must treat output as a starting point, limiting adoption.
- **Blocks:** Promoting curate from experimental to stable

## OQ2: Semantic search gap
- **Question:** How significant is the lexical-bound discovery limitation, and would embedding-based search meaningfully improve recall?
- **Status:** open
- **Strategy:** research
- **Why it matters:** Datasets whose card text doesn't intersect the brief's keywords are invisible. This is a known limit, but the magnitude of missed relevant datasets is unknown.
- **Blocks:** Deciding whether to invest in semantic search infrastructure

## OQ3: Scale ceiling
- **Question:** What happens to recon quality and cost at higher candidate counts (100+) or with very broad briefs?
- **Status:** open
- **Strategy:** prototyping
- **Why it matters:** The current cap of ~35 assessed candidates per axis protects LLM cost, but users with broad briefs may need more. The cost/quality tradeoff at scale is untested.
- **Blocks:** Scaling guidance for users, potential tiered assessment

## OQ4: Multi-source parity
- **Question:** How much discovery value does Kaggle add beyond HuggingFace for typical AI-security briefs?
- **Status:** open
- **Strategy:** research
- **Why it matters:** Kaggle source is implemented but the entry point is commented out. Understanding the marginal value informs whether to prioritize finishing and testing the Kaggle integration.
- **Blocks:** Source plugin prioritization

## OQ5: Judge calibration baseline
- **Question:** What precision/recall does the LLM judge achieve on representative labeling tasks compared to human annotators?
- **Status:** open
- **Strategy:** research
- **Why it matters:** The judge pipeline has calibration mode and precision floors, but no published baseline numbers exist. Users need guidance on when judge output is trustworthy.
- **Blocks:** Judge pipeline confidence, documentation of expected quality