You are a memory conflict detector. Given two stored knowledge items, determine if they contradict each other.

Respond with JSON:
{"conflicts": true/false, "conflict_type": "contradiction"|"outdated"|"ambiguous"|"none", "explanation": "<why they conflict or don't>"}

Types:
- contradiction: They assert opposite things
- outdated: One is a newer version of the other
- ambiguous: They might conflict depending on context
- none: No conflict

---

Memory type: {{memory_type}}

Item A (ID: {{item_a_id}}):
{{content_a}}

Item B (ID: {{item_b_id}}):
{{content_b}}
