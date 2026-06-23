You check for cross-type memory incoherence. Given an SFM fact and related items from other memory types (AM=entities, MM=procedures, EM=preferences), determine if they contradict.

Respond with JSON:
{"incoherent": true/false, "explanation": "<why>", "severity": "low"|"medium"|"high"}

---

SFM Fact (ID: {{fact_id}}):
{{fact_content}}

Related items from other types:
{{related_items}}
