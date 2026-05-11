# LLM hallucinated column names in strategy assessment

**Source:** mcp

The strategy assessor could hallucinate column names or label values that don't exist in the actual dataset, despite being fed 8 real sample rows. This would produce transform specs that fail at curate time or mislead reviewers who trust the report without checking.
