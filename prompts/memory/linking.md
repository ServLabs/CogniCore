You determine relationships between knowledge items from different memory types. Given two items, decide if they are meaningfully related and label the relationship.

Respond with JSON:
{"related": true/false, "type": "<relationship_type>", "confidence": <0.0-1.0>, "direction": "source_to_target"|"target_to_source"|"bidirectional"}

Relationship types: mentions, describes, exemplifies, contradicts, supports, part_of, used_in, related_to, prerequisite_for, depends_on

---

Source [{{source_type}}] (ID: {{source_id}}):
{{source_content}}

Target [{{target_type}}] (ID: {{target_id}}):
{{target_content}}
