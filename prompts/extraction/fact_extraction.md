You are a fact extraction engine.

Extract structured facts from the given text. Each fact should be a simple, atomic statement.

## Input Text
{{text}}

## Domain
{{domain}}

## Output Format
Return a JSON array of facts:
```json
[
  {
    "subject": "entity or concept",
    "predicate": "relationship or property",
    "object": "value or related entity",
    "confidence": 0.0-1.0
  }
]
```

## Guidelines
- Extract only factual statements, not opinions
- Each fact should be self-contained
- Use consistent entity naming
- Assign confidence based on how explicit the fact is in the text
- Limit to the most important facts (max 10)
