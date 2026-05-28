# AI Super Agent — Technical Architecture & Implementation

> **Status**: In Progress — Living Document  
> **Started**: 2026-05-11  
> **Scope**: Super agent with a domain-agnostic framework

---

## 1. Design Philosophy

Build an AI agent that mimics human cognitive patterns — starting with a layered memory system. The framework is domain-agnostic and can be adapted to any domain by configuring the identity, knowledge base, and procedures. 

---

## 2. Memory Layer ✅ COMPLETE

The memory system is modeled after human cognition, with distinct memory types serving different purposes.

### 2.1 Autobiographical Memory (`abm.py`)

**Purpose**: The agent's self-identity — who it is, what it does, its capabilities and constraints. Analogous to a human's self-knowledge and personal identity.

**Key Properties**:
- **Immutable** — Cannot be modified or overwritten at runtime. Prevents identity drift from conversations, prompt injection, or memory poisoning.
- **Always available** — Injected into the system prompt; no retrieval step needed.
- **Dual audience** — Serves both end-users ("Who are you?") and internal AI subsystems ("What am I? What can I do?").

**Implementation**:
- Python `@dataclass(frozen=True)` — enforces immutability at the language level, not just by convention.
- Compact payload (a few hundred tokens) — small enough to include in every LLM call without meaningful token budget impact.
- Serialized to string and injected into the system prompt for every interaction.

**Contents** (schema):
| Field | Description |
|---|---|
| `name` | Agent identity / display name |
| `role` | High-level role description |
| `domain` | Primary domain (e.g., "finance") |
| `capabilities` | List of things the agent can do |
| `constraints` | Boundaries / what it cannot or should not do |
| `version` | Agent version identifier |

**Design Decisions**:
| Decision | Rationale |
|---|---|
| Frozen dataclass over DB/file | Identity is small, static, and read-hot. In-memory is the fastest path. |
| System prompt injection over semantic search | Eliminates retrieval latency and failure modes. For a compact identity payload, always-present beats on-demand. |
| Immutability enforced at code level | Convention-based immutability is fragile. `frozen=True` raises `FrozenInstanceError` on mutation attempts. |

**Future Option**: If identity payload grows (e.g., large capabilities matrix, tool inventory), layer a small in-memory semantic index on top for selective sub-agent queries while keeping a compact core in the system prompt.

---

### 2.2 Prospective Memory (`pm.py`)

**Purpose**: The agent's future-oriented task memory — what it needs to do and when. Analogous to a human's ability to remember to do things at specific times ("pick up groceries at 5 PM", "send report every Monday").

**Key Properties**:
- **Persistent** — Survives restarts. Stored in SQLite (migration path to Postgres).
- **Time-rounded** — Tasks scheduled at 15-minute intervals (:00, :15, :30, :45) for simplicity.
- **Event-driven scheduler** — Sleeps until the next due task; wakes immediately when new tasks are inserted. No wasted polling cycles.
- **Cron support** — Recurring events stored as cron expressions in the schedules table; scheduler materializes next occurrences.
- **Auditable** — Two-table design separates intent (schedules) from outcome (execution log).
- **Eval-ready** — Both tables serve as evaluation datasets: schedule adherence, success rates, failure patterns.

**Architecture**:

```
Task Insert → notify_new_task() → Scheduler wakes
                                      ↓
                              Query: next due task
                                      ↓
                              asyncio.sleep(delta)
                                      ↓
                              Execute task → Log result
                                      ↓
                              Loop: find next due task
```

**Database Schema**:

**Table 1: `schedules`** (Intent — what should happen)

| Column | Type | Description |
|---|---|---|
| `id` | TEXT (UUID) | Primary key |
| `task_name` | TEXT | Human-readable task label |
| `task_type` | TEXT | Category (e.g., "data_pull", "report", "alert") |
| `payload` | JSON | Task parameters / arguments |
| `scheduled_at` | DATETIME | Next scheduled execution (rounded to 15-min) |
| `cron_expr` | TEXT | Cron expression for recurring tasks (NULL = one-time) |
| `priority` | INTEGER | Execution priority (lower = higher priority) |
| `status` | TEXT | `SCHEDULED` / `PAUSED` / `CANCELLED` |
| `created_at` | DATETIME | When the schedule was created |
| `updated_at` | DATETIME | Last modification timestamp |

**Table 2: `execution_log`** (Reality — what actually happened)

| Column | Type | Description |
|---|---|---|
| `id` | TEXT (UUID) | Primary key |
| `schedule_id` | TEXT (FK) | Links back to the schedule that triggered this |
| `scheduled_at` | DATETIME | When it was supposed to run |
| `started_at` | DATETIME | When execution actually started |
| `completed_at` | DATETIME | When execution finished |
| `status` | TEXT | `SUCCESS` / `FAILED` / `SKIPPED` / `TIMEOUT` |
| `result` | JSON | Output / return value from the task |
| `error` | TEXT | Error message / traceback (if failed) |
| `duration_ms` | INTEGER | Execution duration in milliseconds |

**Scheduler Implementation** (event-driven, asyncio):
- `asyncio.Event` used as the wake signal.
- On task insert → call `notify_new_task()` → scheduler re-evaluates next due task.
- On cron task completion → compute next occurrence from `cron_expr`, update `scheduled_at`, log result.
- On one-time task completion → mark schedule `COMPLETED`, log result.

**Eval Use Cases** (metrics written to DuckDB analytics layer):
- **Schedule adherence**: `started_at - scheduled_at` measures latency/drift.
- **Reliability**: Success/failure rates per `task_type`.
- **Performance**: `duration_ms` trends over time.
- **Coverage**: Schedules with no corresponding execution log = missed tasks.

> **Note**: `execution_log` data is appended to DuckDB (`analytics.duckdb`), not stored in SQLite. Schedule definitions remain in `hot.db`.

**Design Decisions**:
| Decision | Rationale |
|---|---|
| SQLite (with Postgres migration path) | Single-agent, file-based, no server overhead. Migrate when multi-agent or distributed. |
| Event-driven over timer loop | Exact timing, instant reaction to new tasks, no wasted polling (~20 lines more code). |
| Two tables over single table | Separates intent from outcome. One schedule → many executions (recurring). Clean audit trail. Natural eval dataset. |
| 15-min rounding | Simplifies scheduling UX and aligns with cron-style intervals. Not a hard limitation — can go finer-grained later. |
| Cron in source table | Recurring events are first-class. Scheduler materializes next run; no external cron daemon needed. |

---

### 2.3 Working Memory (`wm.py`)

**Purpose**: Active conversation context — the agent's "RAM" for ongoing interactions. Stores current conversations, provides fast recall of recent context, and gracefully degrades to summaries for older context.

**Key Properties**:
- **Tiered storage** — Redis (hot/fast) + Local file storage (cold/persistent).
- **Token-aware windowing** — Redis holds recent messages by token count (~50 messages as heuristic), not arbitrary truncation.
- **Rolling summarization** — When buffer hits threshold, summarize (previous summary + new messages → fresh summary). No lossy truncation, no telephone-game degradation.
- **Persist on every summarization** — Summary written to local storage immediately upon creation. Local storage is never more than one window behind Redis, regardless of Redis failure.
- **TTL 24h inactivity** — Redis keys expire after 24h of no reads/writes. Cleanup mechanism, not the persistence trigger.
- **Full audit trail** — Complete conversation stored as JSONL files locally.

**Storage Layout**:

```
Redis (hot):
  wm:{uid}:{convo_id}:messages    → List of recent messages (token-bounded)
  wm:{uid}:{convo_id}:summary     → Latest rolling summary
  TTL: 24h on inactivity (reset on read/write)

Local Storage (cold):
  data/conversations/{uid}/{convo_id}/messages.jsonl   → Full conversation (append-only)
  data/conversations/{uid}/{convo_id}/summary.json     → Latest snapshot summary
```

**Data Flow — Normal Operation**:

```
User message arrives
    ↓
Append to Redis messages list + Append to local JSONL file
    ↓
Check token count of Redis messages
    ↓
If over threshold:
    Feed (previous summary + oldest messages beyond window) to LLM
        ↓
    Produce new rolling summary
        ↓
    Store summary in Redis AND local storage (immediately)
        ↓
    Trim oldest messages from Redis
    ↓
Continue conversation with: [summary] + [recent messages] as context
```

**Data Flow — Reload (Redis miss after TTL expiry)**:

```
User sends message → Redis MISS on wm:{uid}:{convo_id}
    ↓
Load from local storage:
    1. Fetch latest summary.json → inject as context (fast, compact)
    2. Optionally fetch last N messages from JSONL for detail
    ↓
Repopulate Redis → continue as normal
```

**Summarization Strategy**:
- **Rolling summary**: Each summarization takes `previous_summary + new_messages_being_evicted` → produces ONE fresh summary.
- Local `summary.json` is always overwritten with the latest (not a chain of summaries).
- This avoids recursive summarization degradation.

**Design Decisions**:
| Decision | Rationale |
|---|---|
| Redis + local storage over single store | Redis gives sub-ms reads for active conversations. Local files give durable persistence for history. |
| Token-based window over message count | Messages vary wildly in length. Token budget is what the LLM actually cares about. |
| Rolling summary over truncation | Preserves semantic context. Old information is compressed, not lost. |
| Persist summary on every summarization | Local storage is never stale by more than one window. Protects against Redis crashes / unexpected eviction. |
| TTL as cleanup, not persistence trigger | Redis keyspace notifications are unreliable (fire-and-forget). Persistence is handled proactively at summarization time. |
| JSONL for full history | Append-only is fast and audit-friendly. No read-modify-write. |

---

### 2.4 Emotional Memory (`em.py`)

**Purpose**: Tracks user sentiment and communication preferences per interaction. Not empathy simulation — a professional calibration signal for self-correction, audit, and eval.

**Key Properties**:
- **Lean & scoped** — Captures sentiment (frustrated/neutral/satisfied), user preferences (brevity, formality), and nothing more.
- **In-memory current state** — Real-time dict of current session sentiment + preferences for immediate response calibration.
- **Persistent in SQLite** — Same DB as Prospective Memory. Flushed on session end or sentiment shift. Enables JOINs with `execution_log` for root-cause analysis.
- **Professional only** — No empathy acting, no emotional mirroring. Just signal capture and adaptation.
- **Self-correction trigger** — Frustrated signal → log context → analyze what went wrong → adapt behavior.

**In-Memory State** (per active session):

```python
@dataclass
class SessionSentiment:
    user_id: str
    convo_id: str
    current_sentiment: str          # "frustrated" | "neutral" | "satisfied"
    confidence: float               # 0.0 - 1.0
    preferences: dict               # {"response_length": "short", "formality": "casual", ...}
    sentiment_history: list[str]    # track shifts within session
```

**Database Schema** (SQLite — same DB as PM):

**Table: `user_sentiment_log`** (Audit + Eval)

| Column | Type | Description |
|---|---|---|
| `id` | TEXT (UUID) | Primary key |
| `user_id` | TEXT | User identifier |
| `convo_id` | TEXT | Conversation identifier |
| `sentiment` | TEXT | `frustrated` / `neutral` / `satisfied` |
| `confidence` | REAL | Model confidence in the classification (0.0–1.0) |
| `trigger` | TEXT | What caused the sentiment (e.g., "task_failure", "slow_response", "good_result") |
| `context_snippet` | TEXT | Brief excerpt of what was happening when sentiment was detected |
| `timestamp` | DATETIME | When sentiment was recorded |

**Table: `user_preferences`** (Learned over time)

| Column | Type | Description |
|---|---|---|
| `user_id` | TEXT | Primary key |
| `response_length` | TEXT | `short` / `medium` / `detailed` |
| `formality` | TEXT | `casual` / `professional` / `formal` |
| `detail_level` | TEXT | `summary` / `standard` / `deep_dive` |
| `updated_at` | DATETIME | Last preference update |

**Flush Strategy**:
- In-memory state is written to SQL on:
  - Session end (normal close)
  - Sentiment shift (e.g., neutral → frustrated)
  - Explicit threshold (every N messages as fallback)

**Eval Use Cases** (metrics written to DuckDB analytics layer):
- **Frustration root cause**: JOIN sentiment metrics with execution metrics in DuckDB → what broke?
- **Preference drift**: Track how user preferences evolve over time.
- **Agent quality**: Ratio of satisfied vs frustrated sessions over time.
- **Self-correction effectiveness**: After detecting frustration, did the next interaction improve?

> **Note**: Sentiment events are appended to DuckDB (`analytics.duckdb`). User preferences (state) remain in `hot.db`.

**Design Decisions**:
| Decision | Rationale |
|---|---|
| Same SQLite as PM | Simple JOINs with execution_log for correlation. Single migration path to Postgres. |
| In-memory + flush over pure SQL | Real-time calibration needs sub-ms reads. SQL is for persistence/audit, not hot-path. |
| Professional scope only | Users want accuracy, not emotional rapport. Prevents uncanny valley. |
| Sentiment + preferences as separate tables | Sentiment is event-stream (many rows per user). Preferences are state (one row per user, updated). Different access patterns. |

---

### 2.5 Associative Memory (`am.py`)

**Purpose**: The agent's knowledge graph — connects concepts, entities, and relationships to enable multi-hop reasoning and semantic discovery. Analogous to how humans link "billing error" → "system outage" → "migration window" without being explicitly told the chain.

**Key Properties**:
- **Hybrid architecture** — Explicit graph (Kuzu) + implicit semantic index (FAISS).
- **Explicit associations** — Typed, directed relationships between entities (Cypher queries, multi-hop traversal).
- **Implicit associations** — Semantic similarity between concepts even without explicit edges (vector search).
- **Dynamic growth** — New nodes and edges form from conversations, task execution, and data ingestion.
- **Fully embedded** — No servers. Both Kuzu and FAISS run in-process, file-persisted.

**Architecture**:

```
┌─────────────────────────────────────────────┐
│          Associative Memory (am.py)          │
├─────────────────┬───────────────────────────┤
│  Explicit Graph │   Semantic Vector Index   │
│  (Kuzu)         │   (FAISS)                 │
├─────────────────┼───────────────────────────┤
│  Entities       │   Embeddings of entities  │
│  Relationships  │   + descriptions          │
│  Properties     │   Similarity search       │
│  Multi-hop      │   "Find related to X"     │
│  Cypher queries │   even without explicit   │
│                 │   edges                   │
└─────────────────┴───────────────────────────┘
         ↓                    ↓
    "What is directly     "What is semantically
     connected to X?"      similar to X?"
         ↓                    ↓
         └────── MERGED RESULTS ──────┘
```

**Two Types of Association**:

```
EXPLICIT (graph):
  "billing_error" —[caused_by]→ "system_outage"
  "user_123" —[asked_about]→ "churn_analysis"
  "comp_plan_disc" —[contains_column]→ "ACTV_AMT"

IMPLICIT (vector):
  "revenue leakage" ≈ "billing discrepancy"   (semantically near, no explicit link)
  "subscriber churn" ≈ "account cancellation"  (different terms, same concept)
```

**Kuzu Graph Schema** (explicit associations):

```cypher
-- Node types
CREATE NODE TABLE Entity (
    id STRING PRIMARY KEY,
    name STRING,
    type STRING,          -- "dataset", "column", "concept", "user", "task", "error"
    domain STRING,        -- "billing", "collections", "general"
    description STRING,
    created_at TIMESTAMP
)

-- Relationship types
CREATE REL TABLE RELATED_TO (FROM Entity TO Entity, relation STRING, weight FLOAT, created_at TIMESTAMP)
CREATE REL TABLE CAUSED_BY (FROM Entity TO Entity, confidence FLOAT, source STRING)
CREATE REL TABLE CONTAINS (FROM Entity TO Entity)
CREATE REL TABLE ASKED_ABOUT (FROM Entity TO Entity, count INT64, last_at TIMESTAMP)
```

**FAISS Vector Index** (implicit associations):

| Component | Detail |
|---|---|
| Index type | `IndexFlatIP` (inner product) to start; `IndexIVFFlat` at scale |
| Embedding model | TBD (e.g., `text-embedding-3-small` or local model) |
| What's embedded | Entity name + description concatenated |
| Metadata mapping | Parallel list/dict: `faiss_id → entity_id` for lookup back to Kuzu |
| Persistence | `faiss.write_index()` / `faiss.read_index()` to disk |

**Query Flow**:

```
User query: "What affects revenue leakage?"
    ↓
┌──────────────────────────────┐
│ 1. Embed query with same     │
│    model used for entities   │
│ 2. FAISS: top-K similar      │──→ ["billing_discrepancy", "ACTV_AMT", "ADJ credits"]
│ 3. Kuzu: multi-hop from      │
│    matched entities          │──→ "billing_discrepancy" —[caused_by]→ "SOC_mismatch"
│ 4. Merge & rank results      │
└──────────────────────────────┘
    ↓
Enriched context for LLM response
```

**Growth — How Associations Form**:
- **From conversations**: When the agent discusses a topic, extract entities and relationships → insert into Kuzu + embed in FAISS.
- **From task execution**: When a scheduled task runs (PM), log input/output entities and link them.
- **From data ingestion**: When new datasets are loaded, auto-extract schema entities (tables, columns) and link them.
- **Manual curation**: Admin can add/edit associations directly.

**Storage**:

| Component | Location | Persistence |
|---|---|---|
| Kuzu graph | `./data/am_graph/` directory | File-based, survives restarts |
| FAISS index | `./data/am_vectors.index` | `faiss.write_index()` on updates |
| ID mapping | `./data/am_id_map.json` | FAISS ID ↔ Entity ID lookup |

**Design Decisions**:
| Decision | Rationale |
|---|---|
| Kuzu over Neo4j | Embedded, no server/JVM, MIT licensed, Cypher support. Same deployment philosophy as SQLite. Migration path to Neo4j if needed. |
| FAISS over ChromaDB | Leaner, full control, no abstraction overhead. Move to Chroma if metadata management becomes a burden. |
| Hybrid over graph-only | Graph only finds explicit links. Vector finds semantic similarity without explicit edges. Together = GraphRAG-style reasoning. |
| Merged results | Explicit (high confidence, structured) + implicit (discovery, serendipity) gives richer context than either alone. |
| Entity extraction from conversations | Associations grow organically. The graph gets smarter as the agent is used. |

**Growth Path**:
```
Kuzu (now) → Neo4j (multi-agent / distributed / massive scale)
FAISS (now) → Qdrant / Weaviate (managed, distributed vector search)
```

---

### 2.6 Motor Memory (`mm.py`)

**Purpose**: The agent's procedural knowledge — learned, reusable action sequences (both executable code and natural language steps). Analogous to human muscle memory: "I know how to do this, I don't need to re-think every step." Eliminates redundant LLM reasoning for proven workflows.

**Key Properties**:
- **Dual format** — Procedures can be executable Python (`.py`) and/or natural language steps (`.yaml`). Supports both direct execution and LLM-guided reasoning.
- **Dual source** — Admin/developer-authored + system-generated (learned from experience).
- **File-based** — Human-editable, git-versionable, importable by Python, admin-friendly.
- **Indexed** — SQLite metadata index for fast lookup, versioning, usage stats.
- **Semantically discoverable** — FAISS embedding of procedure descriptions for intent-based matching.

**Architecture**:

```
procedures/                          ← Human & system authored
├── billing/
│   ├── revenue_leakage.py           ← Executable Python procedure
│   ├── revenue_leakage.yaml         ← Metadata + NL steps
│   ├── churn_backtest.py
│   └── churn_backtest.yaml
├── collections/
│   ├── aging_report.py
│   └── aging_report.yaml
└── _system_generated/
    ├── auto_001.py                  ← Agent-learned procedures
    └── auto_001.yaml
```

**Three-Layer Storage**:

| Layer | What | Why |
|---|---|---|
| **Files** (`.py` + `.yaml`) | Actual procedure logic + NL steps | Human-editable, git-versionable, importable, admin-friendly |
| **SQLite** (same DB as PM/EM) | Metadata index: name, version, domain, tags, usage count, success rate, file path | Fast lookup, versioning, eval stats, queryable |
| **FAISS** (shared with am.py) | Embedded procedure descriptions | Semantic intent matching: "show me revenue issues" → finds `revenue_leakage` |

**YAML Metadata Format**:

```yaml
name: revenue_leakage_analysis
version: 2
domain: billing
description: "Identifies revenue leakage by comparing expected vs actual charges"
intent_patterns:
  - "revenue leakage"
  - "billing discrepancy"
  - "charge mismatch"
source: admin                    # "admin" | "system_generated"
steps_nl:
  - "Query comp_plan_disc for the target BILL_MONTH"
  - "Filter ACTV_CODE = 'ADJ' and ACTV_AMT < 0"
  - "Aggregate by SOC to find top leakage sources"
  - "AVOID: Do not include BL_IGNORE_IND = 'Y' records"
tags: [billing, revenue, telegence]
created_at: 2026-05-11
updated_at: 2026-05-11
```

**SQLite Schema** (same DB):

**Table: `procedures`**

| Column | Type | Description |
|---|---|---|
| `id` | TEXT (UUID) | Primary key |
| `name` | TEXT | Procedure name (unique) |
| `version` | INTEGER | Version number (increments on update) |
| `domain` | TEXT | Domain (e.g., "billing", "collections") |
| `source` | TEXT | `admin` / `system_generated` |
| `description` | TEXT | Human-readable description |
| `file_path_py` | TEXT | Path to `.py` file (nullable — may be NL-only) |
| `file_path_yaml` | TEXT | Path to `.yaml` file |
| `usage_count` | INTEGER | How many times executed |
| `success_rate` | REAL | Success ratio (0.0–1.0) |
| `last_used_at` | DATETIME | Last execution timestamp |
| `created_at` | DATETIME | When created |
| `updated_at` | DATETIME | Last modification |

**Runtime Flow**:

```
User intent → Embed → FAISS search → match procedure
    ↓
Load YAML (NL steps + metadata) + import .py (executable logic)
    ↓
Execute directly OR feed NL steps to LLM for guided reasoning
    ↓
Log usage + outcome to SQLite (for eval)
    ↓
Update success_rate, usage_count
```

**Procedure Lifecycle**:

```
Admin path:    Dev writes .py + .yaml → git push → agent indexes on startup
System path:   Agent solves new task → extracts steps → writes .py + .yaml
               to _system_generated/ → admin reviews → promotes to domain folder
```

**Design Decisions**:
| Decision | Rationale |
|---|---|
| Files over DB blobs | Code in files = editable in IDE, git-diffable, Python-importable. Admin experience matters. |
| YAML over JSON for metadata | Human-editable, supports multi-line NL steps, comments. Admin-friendly. |
| Shared SQLite | Same DB as PM/EM. JOINable — correlate procedure success with execution_log and sentiment. |
| Separate FAISS index (mm.index) | Procedures have distinct semantic space from general entities. Dedicated index enables procedure-specific tuning and avoids cross-contamination. |
| `_system_generated/` separation | Admin review gate. System-learned procedures don't mix with curated ones until promoted. |
| NL steps alongside code | Some procedures are advisory ("avoid X, prefer Y") not executable. LLM reads YAML steps when reasoning through a task. |

---

### 2.7 Short Form Memory (`sfm.py`)

**Purpose**: The agent's quick-reference cache — frequently accessed facts, compressed summaries, common patterns, and hot knowledge. Analogous to sticky notes on a desk: small, curated, instant recall.

**Key Properties**:
- **Compressed** — Short-line facts, bullet-point summaries, not full documents.
- **Hot cache over LFM** — Most-queried facts from Long Form Memory get promoted here.
- **Semantically searchable** — Dedicated FAISS index over compressed facts for natural language queries.
- **Pinnable** — Admin can force-keep critical facts regardless of access frequency.
- **Frequency-aware** — Auto-promotes from LFM based on access count; LRU eviction for unpinned items.

**Storage**:

| Layer | What | Why |
|---|---|---|
| **In-memory dict** | Runtime hot cache | Fastest possible reads, zero I/O |
| **SQLite** (same DB) | Persistence + metadata (tags, pins, frequency, source) | Survives restarts, queryable |
| **FAISS** (dedicated SFM index) | Semantic search over compressed facts | NL queries find facts even with different phrasing |

**SQLite Schema** (same DB):

**Table: `short_form_facts`**

| Column | Type | Description |
|---|---|---|
| `id` | TEXT (UUID) | Primary key |
| `fact` | TEXT | The compressed fact / summary line |
| `domain` | TEXT | Domain (e.g., "billing", "collections") |
| `source` | TEXT | `promoted_from_lfm` / `admin` / `system_generated` |
| `source_ref` | TEXT | Reference to LFM chunk or document it came from (nullable) |
| `tags` | TEXT (JSON array) | Searchable tags |
| `pinned` | BOOLEAN | Admin-pinned (exempt from eviction) |
| `access_count` | INTEGER | How many times retrieved |
| `last_accessed` | DATETIME | Last retrieval timestamp |
| `created_at` | DATETIME | When created |

**Example Facts**:
```
"ACTV_AMT: positive = charge to subscriber, negative = credit/refund"
"comp_plan_disc grain: BAN + SUBSCRIBER_NO + SOC + ACTV_CODE + Bill Cycle"
"Churn anchor: SUB_STATUS = 'C', date = SUB_STATUS_DATE"
"eTRACS flow: eTRACS → Mainframe → GRID → ADLS → Snowflake"
"Always filter BL_IGNORE_IND = 'Y' (test records)"
```

**Promotion from LFM**:
```
LFM chunk access_count exceeds threshold
    ↓
Extract compressed fact (LLM summarization or admin-curated)
    ↓
Insert into SFM (in-memory + SQLite + FAISS)
    ↓
Set source_ref back to LFM chunk for traceability
```

---

### 2.8 Long Form Memory (`lfm.py`)

**Purpose**: The agent's deep knowledge store — full documents, detailed experiences, ingested domain knowledge, word-for-word context. Analogous to a filing cabinet and library: everything is here, searchable, but retrieval takes longer than SFM.

**Key Properties**:
- **Detailed & complete** — Full documents, schemas, methodologies, past experiences with full context.
- **Semantically searchable** — Dedicated FAISS index over chunked documents (classic RAG).
- **Growing** — New knowledge ingested from admin uploads, learning from experiences, and data catalog updates.
- **Source of truth for SFM** — Frequently accessed chunks get promoted to Short Form Memory.

**Storage**:

| Layer | What | Why |
|---|---|---|
| **Local `.md` files** (`data/` folder) | Raw documents, knowledge articles, schemas | Human-editable, git-versionable, admin-friendly. Backup managed externally. |
| **SQLite** (same DB) | Metadata index: source, domain, tags, access_count, chunk mappings | Fast lookup, tracks what's been ingested, feeds SFM promotion |
| **FAISS** (dedicated LFM index) | Chunked document embeddings | Semantic search for RAG retrieval |

**SQLite Schema** (same DB):

**Table: `long_form_documents`**

| Column | Type | Description |
|---|---|---|
| `id` | TEXT (UUID) | Primary key |
| `title` | TEXT | Document / knowledge article title |
| `file_path` | TEXT | Path to local `.md` file |
| `domain` | TEXT | Domain (e.g., "billing", "collections") |
| `source` | TEXT | `admin_upload` / `ingested` / `learned` |
| `tags` | TEXT (JSON array) | Searchable tags |
| `chunk_count` | INTEGER | Number of chunks extracted |
| `access_count` | INTEGER | Total retrieval hits across all chunks |
| `ingested_at` | DATETIME | When document was ingested |
| `updated_at` | DATETIME | Last modification |

**Table: `lfm_chunks`**

| Column | Type | Description |
|---|---|---|
| `id` | TEXT (UUID) | Primary key |
| `document_id` | TEXT (FK) | Links to `long_form_documents` |
| `chunk_index` | INTEGER | Order within document |
| `content` | TEXT | Chunk text |
| `faiss_id` | INTEGER | Corresponding FAISS vector ID |
| `access_count` | INTEGER | How many times this chunk was retrieved |
| `promoted_to_sfm` | BOOLEAN | Whether a compressed fact was extracted to SFM |

**Tiered Search Flow** (across all memory types):

```
User query → Embed
    ↓
1. SFM FAISS (small, fast) → good match? → return compressed fact
    ↓ miss or low confidence
2. LFM FAISS (larger, deeper) → return detailed chunks
    ↓ optionally
3. AM FAISS (associations, related concepts) → enrich with graph context
    ↓
Merge & rank → build context for LLM
```

**Design Decisions**:
| Decision | Rationale |
|---|---|
| Separate FAISS indexes (SFM / LFM / AM) | Tiered search: fast facts first, deep knowledge as fallback, associations for enrichment. Avoids short facts competing with long chunks in ranking. |
| Local `.md` files for LFM | Admin-friendly, git-versionable, already in use (`data/` folder). External backup managed by user. |
| SFM as L1 cache over LFM | Frequent facts served instantly. LFM only hit on cache miss. Same pattern as CPU L1/L2 cache. |
| Promotion based on access_count | Popular knowledge auto-surfaces. No manual curation needed (but admin pinning available). |
| In-memory + SQLite for SFM | Runtime speed + persistence. Load on startup, serve from memory, flush on update. |

---

### 2.9 Meta Memory (`meta.py`)

**Purpose**: The agent's library card catalog — knows what the agent knows, where it lives, and what's missing. A pointer-based index over all other memory types with semantic search for routing queries to the right memory. Analogous to human metamemory: "I know I know this, and I know where to find it."

**Key Properties**:
- **Pointers only** — No content, no summaries, no relationships. Just short topic labels (2-5 words) pointing to the memory type and reference ID that holds the actual knowledge.
- **Semantically searchable** — Dedicated FAISS index over topic labels. Queries match topics even with different phrasing.
- **Knowledge inventory** — "What do I know?" is a scan of this index. "What don't I know?" is the gap log.
- **Gap detection** — Queries with no match get logged as knowledge gaps with ask count. Actionable signal for what to ingest next.
- **Confidence routing** — Match quality determines whether to return directly, dig deeper, or admit ignorance.
- **Registered during consolidation** — Pointers are NOT registered on every write. Instead, during MML memory management (consolidation phase), the system extracts high-quality "gold data" from recent writes, stores it permanently, and THEN registers pointers to Meta Memory. This avoids polluting the index with low-confidence or transient data.

**Storage**:

| Layer | What | Why |
|---|---|---|
| **In-memory dict** | Runtime pointer lookup | Fastest routing |
| **SQLite** (same DB) | Persistence + gap log | Survives restarts, queryable for gaps/stats |
| **FAISS** (dedicated meta index) | Semantic search over topic labels | NL queries match short labels even with different wording |

**SQLite Schema** (same DB):

**Table: `memory_pointers`** (What I know)

| Column | Type | Description |
|---|---|---|
| `id` | TEXT (UUID) | Primary key |
| `topic` | TEXT | Short topic label (2-5 words, e.g., "billing line items") |
| `memory_type` | TEXT | Target memory: `abm` / `pm` / `wm` / `em` / `am` / `mm` / `sfm` / `lfm` |
| `ref_id` | TEXT | Reference ID or path in the target memory |
| `domain` | TEXT | Domain (e.g., "billing", "collections") |
| `faiss_id` | INTEGER | Corresponding FAISS vector ID |
| `created_at` | DATETIME | When pointer was registered |

**Table: `knowledge_gaps`** (What I don't know)

| Column | Type | Description |
|---|---|---|
| `id` | TEXT (UUID) | Primary key |
| `topic` | TEXT | What was asked about (extracted from query) |
| `first_asked` | DATETIME | When this gap was first detected |
| `last_asked` | DATETIME | Most recent ask |
| `ask_count` | INTEGER | How many times users asked about this |
| `resolved` | BOOLEAN | Whether knowledge was later ingested |
| `resolved_at` | DATETIME | When gap was filled (nullable) |
| `resolved_ref` | TEXT | Pointer to the memory that filled it (nullable) |

**Query Routing Flow**:

```
User query → Embed
    ↓
Meta FAISS search → match topic pointer?
    ↓ YES (high confidence)
    Route to target memory type → retrieve content
    ↓ YES (low confidence)
    Route to target + also search SFM/LFM for backup
    ↓ NO match
    Log to knowledge_gaps → search SFM → LFM → AM (fallback)
    → Respond: "I don't have specific knowledge on X"
```

**Design Decisions**:
| Decision | Rationale |
|---|---|
| Pointers only, no content | Single responsibility. Meta memory routes, other memories store. No duplication. |
| Dedicated FAISS index | Topic labels are very short — different embedding characteristics than SFM facts or LFM chunks. Separate index avoids cross-contamination in ranking. |
| Gap log with ask_count | Turns "I don't know" from a dead end into actionable intelligence. High ask_count gaps = priority ingestion targets. |
| Hybrid routing (rule + semantic) | Identity queries → always ABM (rule). Ambiguous queries → meta FAISS search (semantic). Fast when obvious, smart when unclear. |

---

## Memory Layer — Complete Summary

| # | Memory | Module | Storage | Search | Purpose |
|---|---|---|---|---|---|
| 1 | Autobiographical | `abm.py` | Frozen dataclass (in-memory) | Direct (system prompt) | Agent identity — who it is, what it does |
| 2 | Prospective | `pm.py` | SQLite | SQL queries | Future tasks — what to do and when |
| 3 | Working | `wm.py` | Redis + Local files | Direct (in-context) | Active conversations — current session context |
| 4 | Emotional | `em.py` | In-memory + SQLite | Direct + SQL | User sentiment + preferences |
| 5 | Associative | `am.py` | Kuzu + FAISS | Cypher + semantic | Knowledge graph — relationships + semantic discovery |
| 6 | Motor | `mm.py` | Files + SQLite + FAISS | Semantic | Procedural knowledge — reusable workflows |
| 7 | Short Form | `sfm.py` | In-memory + SQLite + FAISS | Semantic (L1) | Quick-reference facts — compressed, hot |
| 8 | Long Form | `lfm.py` | Local .md + SQLite + FAISS | Semantic (L2) | Deep knowledge — full documents, RAG |
| 9 | Meta | `meta.py` | In-memory + SQLite + FAISS | Semantic (router) | Pointer catalog — knows what I know, routes queries |

## Tech Stack — Finalized

| Component | Choice | License | Purpose |
|---|---|---|---|
| **Language** | Python | — | Primary implementation language |
| **Relational DB (hot)** | SQLite (`hot.db`) | Public domain | PM schedules, Meta pointers, SFM facts, EM user_preferences |
| **Relational DB (cold)** | SQLite (`cold.db`) | Public domain | MM procedures, LFM docs/chunks, Meta knowledge_gaps |
| **Analytics DB** | DuckDB (`analytics.duckdb`) | MIT | Append-heavy metrics, evals, performance numbers across entire agent |
| **In-memory cache** | Redis | BSD-3 | WM hot conversation store (token-windowed, TTL 24h) |
| **Local file storage** | JSONL files | — | WM cold conversation persistence (full history + summaries) |
| **Graph DB** | Kuzu (embedded) | MIT | AM explicit knowledge graph (Cypher queries, multi-hop) |
| **Vector search** | FAISS | MIT | Semantic search — separate indexes for AM, MM, SFM, LFM, Meta |
| **Identity store** | `dataclass(frozen=True)` | — | ABM — immutable agent identity |
| **Knowledge files** | Local `.md` files | — | LFM raw document store (git-versionable, admin-editable) |
| **Procedure files** | `.py` + `.yaml` | — | MM executable logic + NL steps |
| **LLM** | TBD | | Core reasoning engine |
| **Orchestration** | TBD | | Agent framework |

### SQLite Split Strategy

| DB | Path | Tables | Priority | Mode |
|---|---|---|---|---|
| `hot.db` | `/datadrive/sqlite/hot.db` | `schedules`, `memory_pointers`, `short_form_facts`, `user_preferences` | 🔴 Top — high perf, read-heavy | WAL, in-memory cache on startup |
| `cold.db` | `/datadrive/sqlite/cold.db` | `procedures`, `long_form_documents`, `lfm_chunks`, `knowledge_gaps` | 🟡 Medium — rebuildable | WAL, standard |

### FAISS Index Inventory

| Index | Contents | Size | Search Tier |
|---|---|---|---|
| `meta.index` | Topic labels (2-5 words) | Tiny | Router — searched first |
| `sfm.index` | Compressed facts | Small | L1 — fast facts |
| `lfm.index` | Chunked documents | Large (on-disk when big) | L2 — deep knowledge |
| `am.index` | Entity descriptions | Medium | L3 — associations |
| `mm.index` | Procedure descriptions | Small | On-demand — intent matching |

### `/datadrive` — Unified Data Directory

All persistent state writes to `/datadrive/`. Backup this folder to clone/migrate the entire agent.

```
/datadrive/
├── sqlite/
│   ├── hot.db                      ← High-priority: schedules, pointers, facts, prefs
│   └── cold.db                     ← Rebuildable: procedures, LFM docs/chunks, gaps
├── duckdb/
│   └── analytics.duckdb            ← Metrics, evals, performance numbers
├── redis/
│   └── dump.rdb                    ← Redis persistence
├── kuzu/
│   └── graph/                      ← AM knowledge graph
├── faiss/
│   ├── am.index
│   ├── mm.index
│   ├── sfm.index
│   ├── lfm.index
│   └── meta.index
├── jsonl/
│   └── {uid}/{convo_id}/
│       ├── messages.jsonl           ← Full conversation history
│       └── summary.json             ← Latest rolling summary
├── md/
│   └── *.md                        ← LFM knowledge documents
└── yaml/
    ├── {domain}/
    │   ├── *.py                    ← MM executable procedures
    │   └── *.yaml                  ← MM metadata + NL steps
    └── _system_generated/
```

### Codebase Structure

```
CogniCore/
├── README.md
├── ARCHITECTURE.md
│
├── memory/                          ← Memory System (single component)
│   ├── __init__.py
│   ├── types/                       ← 9 Memory Types
│   │   ├── abm.py                  ← Autobiographical
│   │   ├── pm.py                   ← Prospective
│   │   ├── wm.py                   ← Working
│   │   ├── em.py                   ← Emotional
│   │   ├── am.py                   ← Associative
│   │   ├── mm.py                   ← Motor
│   │   ├── sfm.py                  ← Short Form
│   │   ├── lfm.py                  ← Long Form
│   │   └── meta.py                 ← Meta
│   ├── management/                  ← Memory Management Layer
│   │   ├── mml.py                  ← Main orchestrator
│   │   ├── recall.py               ← Recall engine (RRF, MMR, hybrid)
│   │   ├── budget.py               ← LLM budget manager
│   │   ├── learning/               ← 10 Learning Types
│   │   │   ├── reinforced.py
│   │   │   ├── generalization.py
│   │   │   ├── abstraction.py
│   │   │   ├── analogical.py
│   │   │   ├── corrective.py
│   │   │   ├── transfer.py
│   │   │   ├── meta_learning.py
│   │   │   ├── incremental.py
│   │   │   ├── observational.py
│   │   │   └── contrastive.py
│   │   └── maintenance/             ← 7 Background Daemons
│   │       ├── consolidation.py
│   │       ├── forgetting.py
│   │       ├── conflict.py
│   │       ├── coherence.py
│   │       ├── optimization.py
│   │       ├── integrity.py
│   │       └── linking.py
│   └── procedures/                  ← Motor Memory procedure files
│       ├── billing/
│       │   ├── revenue_leakage.py
│       │   └── revenue_leakage.yaml
│       ├── collections/
│       └── _system_generated/
│
├── analytics/
│   └── analytics.py                ← DuckDB metrics + evals
│
└── agent/                           ← Agent Framework (TBD)
```

---

## 3. Analytics Layer (`analytics.py`)

**Purpose**: Cross-cutting metrics and performance tracking across the entire super agent. Not a memory type — an infrastructure layer. Append-only numerical data for admin/developer dashboards.

**Key Properties**:
- **Append-only** — Every metric is an immutable event. No updates, no deletes.
- **Numbers-focused** — Latencies, counts, rates, scores. Not logs or text.
- **Write-heavy, read-light** — Constant appends from all agent components. Admins read dashboards periodically.
- **DuckDB** — Columnar, analytical, single-file, no server. Built for this exact pattern.
- **Read cache** — Dashboard queries cached for 5 minutes. No hot-path reads.

**What gets tracked** (examples):

| Category | Metrics |
|---|---|
| **Usage** | Unique users, sessions/day, messages/session |
| **Memory performance** | Search latency per index (meta/sfm/lfm/am/mm), hit rate, miss rate |
| **Tiered search** | Short-circuit rate, L1 vs L2 vs L3 hit distribution |
| **Task execution** | Scheduled vs executed, success/failure rate, duration_ms |
| **Sentiment** | Frustrated/neutral/satisfied distribution, sentiment shift rate |
| **Knowledge** | Gap count, gap resolution rate, SFM promotion rate |
| **Procedures** | Usage count per procedure, success rate per procedure |
| **System** | LLM call count, token usage, LLM latency, total request latency |

**DuckDB Schema**:

**Table: `metrics`**

| Column | Type | Description |
|---|---|---|
| `ts` | TIMESTAMP | Event timestamp |
| `component` | VARCHAR | Source: `pm` / `wm` / `em` / `am` / `mm` / `sfm` / `lfm` / `meta` / `llm` / `system` |
| `metric_name` | VARCHAR | e.g., `search_latency_ms`, `unique_users`, `hit_rate` |
| `metric_value` | DOUBLE | The numerical value |
| `dimensions` | JSON | Optional tags: `{"index": "sfm", "user_id": "u123"}` |

**Table: `eval_scores`**

| Column | Type | Description |
|---|---|---|
| `ts` | TIMESTAMP | Eval timestamp |
| `eval_type` | VARCHAR | `schedule_adherence` / `search_quality` / `response_quality` / `procedure_success` |
| `score` | DOUBLE | 0.0–1.0 |
| `details` | JSON | Optional breakdown |

**Write pattern**: Fire-and-forget append from any component. Batch buffered in-memory, flushed to DuckDB every N seconds.

**Read pattern**: Admin dashboard queries cached 5 min. Typical queries:
```sql
SELECT component, metric_name, AVG(metric_value)
FROM metrics WHERE ts > now() - INTERVAL 1 HOUR
GROUP BY component, metric_name
```

**Design Decisions**:
| Decision | Rationale |
|---|---|
| DuckDB over SQLite | Columnar = fast analytical aggregations. Write-heavy append pattern is DuckDB's sweet spot. Single file on disk. |
| Separate from memory DBs | Analytics is infrastructure, not memory. Different access pattern, different priority. |
| Append-only, no updates | Immutable event stream. Simple, auditable, no contention. |
| 5-min read cache | Admins don't need real-time. Eliminates read load on DuckDB. |
| Batch writes | Buffer in-memory, flush periodically. Avoids per-event I/O. |

---

## 4. Memory Management Layer (`mml.py`) ✅ COMPLETE

**Purpose**: The agent's subconscious — autonomous processes that maintain, optimize, and evolve the memory system. Responsible for intelligent recall, learning from experience, self-healing, and background maintenance. This is what makes the memory layer "alive."

**Key Properties**:
- **Autonomous** — Runs continuously, acts by itself without user commands.
- **Dual mode** — Real-time (inline with requests) + background ("sleep mode" for heavy lifting).
- **Self-scheduling** — Uses Prospective Memory (PM) to schedule its own background tasks. The memory system manages itself.
- **LLM budget-aware** — Background tasks have a per-cycle budget cap, priority queue, and tiered model selection (expensive model for critical tasks, cheaper for routine).

---

### 4.1 Real-Time Operations (inline with requests)

These run during every user interaction — must be fast.

| Operation | What It Does | Triggered By |
|---|---|---|
| **Memory Recall** | Tiered search across all 9 memory types. Confidence scoring, ranking, multi-hop reasoning. | Every user query |
| **Auto-Registration** | Every memory write → extract topic label → register pointer in Meta Memory. | Any memory write (SFM, LFM, AM, MM) |
| **Light Learning** | Extract obvious facts from conversation inline (e.g., user corrects a fact → update SFM). | Conversation flow |
| **Conflict Detection** | Flag contradictions when detected (don't resolve inline — queue for background). | Retrieval returns conflicting results |
| **Relevance Boosting** | Increment access counts, update `last_accessed` timestamps on retrieved memories. | Every successful retrieval |

### Memory Recall Engine

The most critical real-time operation. Multi-strategy retrieval:

```
User query → Embed
    ↓
1. Meta FAISS → "Do I know about this?" → route to specific memory
    ↓ high confidence → go direct
    ↓ low confidence → tiered search
2. SFM FAISS → compressed facts (L1)
3. LFM FAISS → detailed chunks (L2)
4. AM FAISS + Kuzu → associations + graph traversal (L3)
5. MM FAISS → matching procedures (on-demand)
    ↓
Merge & rank by:
  - Confidence score (embedding similarity)
  - Source priority (SFM > LFM > AM for factual queries)
  - Recency (recent memories weighted higher)
  - Access frequency (popular = likely relevant)
  - Relevance decay (old + unused = downweighted)
    ↓
Build context for LLM
```

---

### 4.2 Background Operations ("Sleep Mode")

Heavy lifting that runs during off-peak hours. The agent doesn't need to operate at full capacity 24/7 — just like human sleep consolidates memories.

**Scheduled via PM cron jobs** — the memory system manages itself using its own scheduler:

```
PM cron: "0 1 * * *"    → Consolidation (1 AM)
PM cron: "0 2 * * *"    → Experience generalization (2 AM)
PM cron: "0 3 * * *"    → Forgetting / pruning (3 AM)
PM cron: "0 4 * * *"    → Coherence scan + conflict resolution (4 AM)
PM cron: "0 5 * * *"    → Index optimization + integrity check (5 AM)
PM cron: "*/30 * * * *" → Auto-registration sweep (every 30 min, lightweight)
```

#### 4.2.1 Memory Consolidation

**What**: Process recent conversations and interactions → extract entities → link in AM graph → create SFM facts → register in Meta.

**How**:
```
Today's conversations (from WM/JSONL)
    ↓
LLM: Extract entities, relationships, key facts
    ↓
Insert entities into Kuzu (AM) → create/strengthen edges
    ↓
Compress key facts → insert into SFM
    ↓
Register all new entries in Meta
    ↓
Log consolidation metrics to DuckDB
```

**Human analogy**: Deep sleep — organizing today's experiences into long-term storage.

#### 4.2.2 Memory Evolution

**What**: Facts get updated, refined, corrected over time. When new information contradicts or extends existing knowledge, evolve the memory.

**How**:
- Track fact versions in SFM/LFM
- When a fact is corrected, update the current version but keep history
- Strengthen associations that keep proving useful, weaken ones that don't

#### 4.2.3 Learning from Experiences

**What**: Generalize patterns from repeated specific events → create reusable procedures (MM) or facts (SFM).

**How**:
```
Scan execution_log (DuckDB): find repeated similar tasks
    ↓
LLM: "I solved 5 similar billing queries this week. What's the common pattern?"
    ↓
Generate procedure (.py + .yaml) → save to MM/_system_generated/
    ↓
Register in Meta → flag for admin review
```

#### 4.2.4 Reinforced Learning

**What**: Track which retrievals led to good outcomes (user satisfied, task succeeded) vs bad outcomes (user frustrated, task failed). Strengthen successful paths.

**How**:
- Correlate retrieval logs with sentiment (EM) and execution results (DuckDB)
- Successful retrieval → boost access_count, strengthen AM edges
- Failed retrieval → flag source memory for review, weaken edges
- Over time: recall engine prioritizes paths that historically worked

#### 4.2.5 Memory Summarization & Promotion

**What**: Compress verbose LFM content → promote to SFM as compressed facts.

**How**:
```
Scan LFM chunks: access_count > threshold AND not yet promoted
    ↓
LLM: Summarize chunk into 1-2 line fact
    ↓
Insert into SFM → set source_ref → mark chunk as promoted
```

#### 4.2.6 Memory Forgetting

**What**: Prune stale, irrelevant, never-accessed, or superseded memories. Keep the system lean.

**Rules**:
- SFM: Evict unpinned facts with zero access in last N days (LRU)
- LFM chunks: Flag for review if zero access in 90 days
- AM edges: Remove edges with zero traversals + low weight
- Meta pointers: Remove if target memory was deleted
- **Never forget**: Pinned items, ABM identity, active PM schedules

**Safeguards**:
- Soft delete first (mark as `archived`) → hard delete after 30 days
- Log all forgetting decisions to DuckDB for audit

#### 4.2.7 Memory Conflict Resolution

**What**: When two memories contradict (SFM says X, LFM says Y), detect, evaluate, and resolve.

**How**:
```
Coherence scan: compare SFM facts against LFM source documents
    ↓
Detect contradiction → create conflict record
    ↓
Resolution strategy:
  1. Source authority: LFM (original doc) > SFM (derived fact) → update SFM
  2. Recency: newer information wins (if both are primary sources)
  3. Admin escalation: flag for human review if ambiguous
    ↓
Log resolution to DuckDB
```

#### 4.2.8 Cross-Memory Linking

**What**: Auto-detect that a new SFM fact relates to an AM entity or an MM procedure — create associations automatically.

**How**: After any memory write, embed the content → search AM for similar entities → create edges if similarity > threshold.

#### 4.2.9 Memory Optimization

**What**: Technical maintenance of storage and indexes.

| Task | Action |
|---|---|
| FAISS rebuild | Remove deleted vectors, re-cluster (IVF), compact |
| Kuzu maintenance | Remove orphan nodes, recalculate edge weights |
| SQLite optimization | `PRAGMA optimize`, `VACUUM` if needed |
| Cache warming | Pre-load hot.db tables into in-memory dicts on startup |
| Embedding refresh | Re-embed entries when embedding model is upgraded |

#### 4.2.10 Memory Integrity

**What**: Self-healing — detect corruption, orphan records, broken references.

| Check | Action |
|---|---|
| SFM fact → deleted LFM chunk | Remove SFM fact or re-derive from document |
| Meta pointer → deleted memory | Remove pointer |
| FAISS ID → no matching SQLite record | Remove vector from index |
| SQLite record → no FAISS vector | Re-embed and insert |

---

### 4.3 LLM Budget Management (Background Tasks)

Background tasks consume LLM calls. Managed with a budget system:

**Budget Cap**: Max N LLM calls per sleep cycle (configurable, e.g., 500 calls/night).

**Priority Queue** — tasks executed in priority order until budget exhausted:

| Priority | Task | Model Tier | Typical Cost |
|---|---|---|---|
| 🔴 P0 | Conflict resolution | Expensive (high reasoning) | ~5 calls per conflict |
| 🔴 P0 | Critical consolidation (today's conversations) | Expensive | ~10-20 calls |
| 🟡 P1 | SFM promotion (summarization) | Cheap (simple compression) | ~1 call per fact |
| 🟡 P1 | Experience generalization | Expensive | ~5-10 calls per pattern |
| 🟢 P2 | Cross-memory linking | Cheap (embedding only, no LLM) | 0 LLM calls |
| 🟢 P2 | Coherence scan | Cheap (comparison) | ~1 call per check |
| 🟢 P2 | Gap analysis | Cheap | ~1 call |

**Model Tiering**:

| Tier | Use Case | Model (example) |
|---|---|---|
| **Expensive** | Reasoning, conflict resolution, generalization, complex extraction | GPT-4o / Claude Sonnet |
| **Cheap** | Summarization, simple extraction, comparison, embedding | GPT-4o-mini / local model |
| **Free** | Embedding, FAISS operations, graph traversal, SQL | No LLM — pure computation |

**Budget Enforcement**:
```python
budget = BudgetManager(max_calls=500, max_cost_usd=5.00)

for task in priority_queue:
    if not budget.can_afford(task.estimated_cost):
        break  # defer remaining to next cycle
    result = task.execute(model=task.model_tier)
    budget.deduct(result.actual_cost)
    log_to_duckdb(task, result)
```

---

### 4.4 Design Decisions

| Decision | Rationale |
|---|---|
| Dual mode (real-time + sleep) | Real-time must be fast (<100ms). Heavy lifting deferred to off-peak. Just like humans. |
| Self-scheduling via PM | No external scheduler needed. The agent's own prospective memory runs maintenance tasks. Elegant recursion. |
| LLM budget cap | Prevents runaway costs in background. Forces prioritization. |
| Model tiering | Critical tasks get best model. Routine tasks use cheap/free. Optimizes cost/quality trade-off. |
| Priority queue | P0 tasks always run. P2 tasks only if budget remains. Guarantees critical maintenance. |
| Soft delete before hard delete | Safety net. Prevents irreversible data loss from forgetting algorithm bugs. |
| Coherence scan as background | Too expensive for real-time. Periodic batch scan catches drift without blocking requests. |

---

### 4.5 Algorithms, Mathematics & State-of-the-Art Techniques

The intellectual backbone of the Memory Management Layer. Most algorithms are LLM-free — pure math/computation — keeping background costs low.

---

#### 4.5.1 Memory Recall — Ranking & Fusion

**Problem**: Multiple signals (embedding similarity, recency, frequency, source priority, relevance decay) — how to combine into one score?

**Algorithm: Reciprocal Rank Fusion (RRF) + Maximal Marginal Relevance (MMR)**

RRF fuses multiple ranked lists without needing score normalization:

$$score(doc) = \sum_{i} \frac{w_i}{k + rank_i(doc)}$$

Where each signal (similarity, recency, frequency, etc.) produces its own ranking, and RRF merges them.

MMR penalizes redundancy in retrieved results to avoid near-duplicate facts:

$$MMR = \arg\max_{d \in R \setminus S} \left[ \lambda \cdot sim(q, d) - (1-\lambda) \cdot \max_{d' \in S} sim(d, d') \right]$$

| Component | Algorithm | LLM Required? |
|---|---|---|
| Multi-signal fusion | Reciprocal Rank Fusion (RRF) | No |
| Redundancy removal | Maximal Marginal Relevance (MMR) | No |
| Keyword + dense hybrid | BM25 sparse + FAISS dense | No |
| Complex query handling | Query decomposition into sub-queries | Yes (cheap) |

**BM25 + Dense Hybrid**: Combine sparse keyword match (BM25) with dense FAISS vector search. Catches exact keyword matches that embeddings miss (e.g., "ACTV_AMT" as a column name).

**Query Decomposition** (out-of-the-box): For complex queries, break into sub-queries, retrieve separately, merge:
```
"What caused revenue leakage in March billing?"
    → sub-query 1: "revenue leakage" → SFM/LFM
    → sub-query 2: "March billing" → LFM (time-filtered)
    → sub-query 3: "cause" → AM (graph traversal from leakage entity)
    → merge results with RRF
```

---

#### 4.5.2 Memory Forgetting — Ebbinghaus Decay Curve

**Problem**: What to forget, when, and how aggressively?

**Algorithm: Modified Ebbinghaus Forgetting Curve with Spaced Repetition**

$$R(t) = e^{-t/S}$$

Where:
- $R(t)$ = retention strength (0 to 1)
- $t$ = time since last access (days)
- $S$ = stability — increases with each retrieval (spaced repetition effect)

```python
def retention_score(last_accessed: datetime, access_count: int, pinned: bool) -> float:
    if pinned:
        return 1.0
    t = (now() - last_accessed).total_seconds() / 86400  # days
    S = base_stability * math.log1p(access_count)  # more access = slower decay
    return math.exp(-t / S)
```

**Spaced repetition effect**: Each access increases $S$ (stability), so frequently-used memories decay slower — exactly like human memory strengthening through rehearsal.

**Competitive Forgetting** (out-of-the-box): Memories compete for slots. SFM has a max capacity. New fact enters → compute retention scores for all → lowest score gets evicted. Naturally keeps the most relevant facts alive.

```python
def evict_if_needed(sfm_facts: list, max_capacity: int, new_fact: Fact):
    if len(sfm_facts) < max_capacity:
        sfm_facts.append(new_fact)
        return
    scores = [(f, retention_score(f.last_accessed, f.access_count, f.pinned)) for f in sfm_facts]
    weakest = min(scores, key=lambda x: x[1])
    new_score = retention_score(now(), 0, False)
    if new_score > weakest[1]:
        soft_delete(weakest[0])
        sfm_facts.append(new_fact)
```

---

#### 4.5.3 Relevance Decay — Time-Weighted Scoring with Category Priors

**Problem**: Old memories shouldn't compete equally with fresh ones. But some old memories are timeless.

**Algorithm: Bayesian Time-Decay with Category Floors**

$$score_{final} = score_{similarity} \times D(t) \times F(n)$$

Where:
- $D(t) = \alpha + (1-\alpha) \cdot e^{-\lambda t}$ — time decay with floor $\alpha$ (never goes to zero)
- $F(n) = 1 - e^{-\beta n}$ — frequency boost (asymptotic, diminishing returns)
- $\alpha$ = category-specific floor

```python
def time_decay(last_accessed: datetime, category: str) -> float:
    floors = {"definition": 0.8, "procedure": 0.6, "event": 0.2, "metric": 0.1}
    alpha = floors.get(category, 0.4)
    t_days = (now() - last_accessed).days
    lambda_ = 0.01  # slow decay
    return alpha + (1 - alpha) * math.exp(-lambda_ * t_days)

def frequency_boost(access_count: int) -> float:
    beta = 0.1
    return 1 - math.exp(-beta * access_count)
```

**Key insight**: Definitions ("ACTV_AMT = monetary amount") should barely decay ($\alpha = 0.8$). Events ("March billing error") should decay fast ($\alpha = 0.2$). The floor per category handles this.

---

#### 4.5.4 Memory Consolidation — Community Detection + Graph Embedding

**Problem**: Raw entities and facts from today's conversations are scattered. How to organize them into the knowledge graph meaningfully?

**Algorithms**:

| Algorithm | Where | What |
|---|---|---|
| **Louvain Community Detection** | AM (Kuzu graph) | Auto-discover clusters of related entities. "These 8 entities all relate to billing adjustments" |
| **Node2Vec** | AM graph → embeddings | Generate graph-aware embeddings that capture structural relationships, not just text similarity |
| **Temporal Knowledge Graph Embedding (TTransE)** | AM graph with time | Relationships have time validity. "X caused Y in March" ≠ "X caused Y always" |

**Consolidation algorithm**:
```
1. Extract entities from today's conversations (LLM: NER + relation extraction)
2. For each entity:
   a. Fuzzy match against existing AM nodes (embedding similarity > 0.9 → same entity)
   b. If new → create node + embed + register in Meta
   c. If existing → strengthen edges, update properties
3. Run Louvain on updated subgraph → discover new communities
4. Community labels → auto-generate SFM summary facts
   "Billing adjustments cluster: ACTV_CODE=ADJ, DISCOUNT_CD, PROMO_ID are related"
```

**Memory Replay** (out-of-the-box — inspired by hippocampal replay in neuroscience): During consolidation, "replay" today's most important interactions through the graph to strengthen paths that were used, exactly like the brain replays experiences during sleep:

```python
for interaction in todays_top_interactions:
    path = trace_retrieval_path(interaction)  # which memories were accessed?
    for edge in path:
        edge.weight *= reinforcement_factor  # strengthen the path
```

---

#### 4.5.5 Conflict Resolution — Truth Discovery + NLI

**Problem**: SFM says "ACTV_AMT is always positive for charges" but LFM has a case where it's negative for a specific adjustment type. Which is right?

**Conflict Detection — Natural Language Inference (NLI)**:

Uses a small local cross-encoder model (no LLM calls, ~1000 checks/second on CPU):

```python
from sentence_transformers import CrossEncoder
nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-small')

def detect_contradiction(fact_a: str, fact_b: str) -> bool:
    # scores = [contradiction, entailment, neutral]
    scores = nli_model.predict([(fact_a, fact_b)])
    return scores[0][0] > 0.7  # high contradiction confidence
```

**Resolution — Source Authority + Corroboration Scoring**:

```
Conflict detected between memory A and memory B
    ↓
Score both by:
  1. Source authority:
     primary_doc (1.0) > admin_created (0.9) > llm_derived (0.7) > conversation_extract (0.5)
  2. Recency: newer source gets bonus
  3. Corroboration: how many other memories support each version?
    ↓
Combined score:
  authority_weight * authority + recency_weight * recency + corroboration_weight * corroboration
    ↓
Winner confidence > threshold → auto-resolve, archive loser
Winner confidence < threshold → escalate to admin
```

**Contradiction-as-Signal** (out-of-the-box): Don't just resolve conflicts — mine them. Contradictions often reveal nuance. "ACTV_AMT is positive for charges" and "ACTV_AMT can be negative" are BOTH true — the original fact was incomplete. The resolution should produce a BETTER, more complete fact: "ACTV_AMT: positive = charge, negative = credit/adjustment."

---

#### 4.5.6 Reinforced Learning — Contextual Multi-Armed Bandit

**Problem**: Which retrieval strategy/path works best for which type of query?

**Algorithm: LinUCB (Contextual Bandit)**

$$a^* = \arg\max_a \left( \hat{\theta}_a^T x + \alpha \sqrt{x^T A_a^{-1} x} \right)$$

Where:
- $x$ = query feature vector (domain, type, length, user history)
- $\hat{\theta}_a$ = learned weights for arm (retrieval strategy) $a$
- $\alpha$ = exploration bonus (try less-used paths occasionally)

**Arms** (retrieval strategies):

| Arm | Strategy | Best For |
|---|---|---|
| 1 | SFM-only | Quick factual lookups |
| 2 | LFM-only | Deep document retrieval |
| 3 | SFM → LFM cascade | General queries |
| 4 | AM graph → LFM | Root-cause analysis, relationship questions |
| 5 | MM procedure lookup → execute | Repeated known tasks |

**Reward signal** (from Emotional Memory + execution results):

| Signal | Reward |
|---|---|
| User satisfied (EM) | +1.0 |
| Task succeeded | +0.5 |
| Task failed | -0.5 |
| User frustrated (EM) | -1.0 |

```python
class LinUCB:
    def __init__(self, n_arms: int, n_features: int, alpha: float = 0.5):
        self.A = [np.eye(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]
        self.alpha = alpha

    def select_arm(self, x: np.ndarray) -> int:
        scores = []
        for a in range(len(self.A)):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            ucb = theta @ x + self.alpha * np.sqrt(x @ A_inv @ x)
            scores.append(ucb)
        return int(np.argmax(scores))

    def update(self, arm: int, x: np.ndarray, reward: float):
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
```

Over time, the agent learns: "For billing column questions → SFM-only works best. For root-cause analysis → AM graph → LFM cascade is optimal."

**Simpler alternative**: **Thompson Sampling** — maintains a Beta distribution per arm, samples from it. Simpler math, still balances exploration vs exploitation.

---

#### 4.5.7 Experience Generalization — Pattern Mining

**Problem**: The agent solved 5 similar queries. How to detect the pattern and create a reusable procedure?

**Algorithms**:

| Step | Algorithm | Purpose |
|---|---|---|
| Clustering | **DBSCAN** over interaction embeddings | Find natural clusters without specifying K |
| Sequence alignment | **Longest Common Subsequence (LCS)** | Extract common action sequence from clustered interactions |
| Distillation | **LLM summarization** | Generate reusable .py + .yaml from examples |

```
Step 1: Cluster similar past interactions
  - Embed each (query, response, action_trace) tuple
  - DBSCAN(eps=similarity_threshold, min_samples=5)
  - Cluster of 5+ similar interactions → candidate pattern
  - Noise points (unique queries) are ignored

Step 2: Extract common action sequence
  - Align action traces from clustered interactions
  - LCS of tool/memory calls across all traces
  - "All 5 queries did: search_SFM → query_LFM → filter_by_date → aggregate → format"

Step 3: LLM distillation
  - Feed the 5 examples + common sequence to LLM (expensive model)
  - "Generate a reusable procedure with parameters"
  - Output: .py + .yaml → save to MM/_system_generated/
  - Register in Meta → flag for admin review
```

---

#### 4.5.8 Memory Summarization — Information-Theoretic Compression

**Problem**: Compress a verbose LFM chunk into an SFM fact without losing critical information.

**Algorithm: Information Density Scoring + Abstractive Summarization**

$$ID(s) = -\log P(s \mid context)$$

Higher information density = more surprising/informative sentence.

```
Step 1: Score each sentence by information density
  - TF-IDF surprise: sentences with rare terms score higher
  - Embedding distance: sentences far from document centroid are more distinctive
  - Perplexity: high perplexity = high information content

Step 2: Extract top-K most informative sentences (extractive)

Step 3: LLM abstractive summarization of extracted sentences → 1-2 line fact (cheap model)
```

```python
def information_density(sentence: str, corpus_tfidf, doc_centroid: np.ndarray) -> float:
    # TF-IDF surprise component
    tfidf_score = np.mean(corpus_tfidf.transform([sentence]).toarray())

    # Embedding distance from centroid
    sent_embed = embed(sentence)
    dist_score = 1 - cosine_similarity(sent_embed, doc_centroid)

    return 0.5 * tfidf_score + 0.5 * dist_score
```

**Hierarchical Summarization** (out-of-the-box): Don't just compress one chunk. Use the AM graph to find related chunks first, then jointly summarize a cluster of related chunks into one SFM fact. Produces richer, more connected facts.

---

#### 4.5.9 Cross-Memory Linking — Link Prediction

**Problem**: Auto-detect that a new SFM fact should be connected to AM entities.

**Algorithm: Cosine + Jaccard Hybrid Scoring**

$$P(link) = \omega \cdot cos(z_u, z_v) + (1-\omega) \cdot J(tokens_u, tokens_v)$$

Where $\omega = 0.7$ (favor semantic over lexical).

```python
def should_link(new_entity: str, existing_entity: str, threshold: float = 0.7) -> bool:
    # Semantic similarity (embedding cosine)
    sem_score = cosine_similarity(embed(new_entity), embed(existing_entity))

    # Keyword overlap (Jaccard index)
    tokens_a = set(tokenize(new_entity.lower()))
    tokens_b = set(tokenize(existing_entity.lower()))
    jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b) if tokens_a | tokens_b else 0

    combined = 0.7 * sem_score + 0.3 * jaccard
    return combined > threshold
```

If `should_link()` → create edge in Kuzu AM graph. No LLM required.

---

#### 4.5.10 Coherence Checking — Entailment Graph

**Problem**: Detect when the agent's knowledge has drifted into inconsistency across memory types.

**Algorithm: Batch NLI Entailment Scan** (local model, no LLM, ~1000 checks/sec on CPU)

```
For each SFM fact:
    Find source LFM chunk (via source_ref)
    Run NLI: does LFM chunk ENTAIL the SFM fact?

    ENTAILS    → coherent ✅
    CONTRADICTS → conflict → queue for resolution ⚠️
    NEUTRAL     → SFM fact may be outdated or overgeneralized → flag for review
```

```python
# Batch NLI — fast, free, no LLM calls
nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-small')

def coherence_scan(sfm_facts: list, lfm_chunks: dict) -> list:
    conflicts = []
    pairs = []
    for fact in sfm_facts:
        if fact.source_ref and fact.source_ref in lfm_chunks:
            pairs.append((fact.text, lfm_chunks[fact.source_ref]))

    results = nli_model.predict(pairs, batch_size=64)
    for i, (contradiction, entailment, neutral) in enumerate(results):
        if contradiction > 0.7:
            conflicts.append({"fact": pairs[i][0], "source": pairs[i][1], "type": "contradiction"})
        elif neutral > 0.7:
            conflicts.append({"fact": pairs[i][0], "source": pairs[i][1], "type": "drift"})
    return conflicts
```

---

### 4.6 Algorithm Stack — Complete Summary

| # | Feature | Algorithm(s) | Type | LLM? | Cost |
|---|---|---|---|---|---|
| 1 | Recall ranking | RRF + MMR | Math | No | Free |
| 2 | Recall hybrid search | BM25 sparse + FAISS dense | Algorithm | No | Free |
| 3 | Query decomposition | Sub-query splitting | LLM | Yes | Cheap |
| 4 | Forgetting | Ebbinghaus decay curve ($e^{-t/S}$) | Math | No | Free |
| 5 | Competitive eviction | Min-retention eviction | Algorithm | No | Free |
| 6 | Relevance decay | Bayesian time-decay with category floors | Math | No | Free |
| 7 | Consolidation clustering | Louvain community detection | Algorithm | No | Free |
| 8 | Graph embeddings | Node2Vec | Algorithm | No | Free |
| 9 | Memory replay | Path reinforcement | Algorithm | No | Free |
| 10 | Conflict detection | NLI cross-encoder (DeBERTa) | Local model | No | Free |
| 11 | Conflict resolution | Source authority + corroboration | Algorithm | Sometimes | Expensive |
| 12 | Retrieval optimization | LinUCB contextual bandit | Math | No | Free |
| 13 | Experience generalization | DBSCAN + LCS + LLM distillation | Algorithm + LLM | Yes | Expensive |
| 14 | Summarization | Information density + abstractive | Math + LLM | Yes | Cheap |
| 15 | Cross-memory linking | Cosine + Jaccard hybrid | Math | No | Free |
| 16 | Coherence checking | Batch NLI entailment | Local model | No | Free |

**Key insight**: 12 of 16 algorithms are completely LLM-free. Only 4 require LLM calls (2 cheap, 2 expensive). This keeps background "sleep mode" costs minimal while delivering sophisticated memory management.

**Additional local models required** (no API cost, runs on CPU):

| Model | Purpose | Size |
|---|---|---|
| `cross-encoder/nli-deberta-v3-small` | Conflict detection + coherence checking | ~140MB |
| BM25 index (rank_bm25 or similar) | Sparse keyword search for hybrid recall | Lightweight |
| Embedding model (pluggable, default: `BAAI/bge-large-en-v1.5`) | FAISS embeddings for all indexes | ~1.3GB |

> **Note**: Embedding model is pluggable via `EMBEDDING_MODEL` env var. Default is `BAAI/bge-large-en-v1.5` (1024 dims) for production quality. Alternatives: `text-embedding-3-small` (OpenAI API, 1536 dims), `nomic-embed-text-v1.5` (768 dims), or any sentence-transformers compatible model. Dimension is auto-detected from model.

---

### 4.7 Learning Taxonomy

The agent has 10 distinct learning types. Each targets specific memory types and uses different algorithms. Together, they create a comprehensive learning system that improves the agent autonomously over time.

---

#### 4.7.1 Reinforced Learning

**What**: Track which retrieval paths led to good vs bad outcomes. Strengthen successful paths, weaken failed ones.

**Algorithm**: LinUCB contextual bandit (see 4.5.6).

**Trigger**: Every user interaction — reward signal from EM (sentiment) + task outcome.

**Memory Target**: Bandit weights (in-memory, persisted to `/datadrive/`).

**LLM**: No.

---

#### 4.7.2 Experience Generalization

**What**: Detect repeated similar tasks → extract the common pattern → create a reusable procedure.

**Algorithm**: DBSCAN clustering + Longest Common Subsequence + LLM distillation (see 4.5.7).

**Trigger**: Sleep mode — scan recent execution history for clusters.

**Memory Target**: MM (new `.py` + `.yaml` in `_system_generated/`).

**LLM**: Yes (expensive — pattern extraction + procedure generation).

---

#### 4.7.3 Abstraction Learning

**What**: Move from specific instances to general concepts. Build a hierarchy of understanding.

**Levels**:
```
Level 0 (specific):  "BAN 12345 had ADJ for SOC mismatch on 2026-03-15"
Level 1 (pattern):   "ADJ activity often correlates with SOC mismatches"
Level 2 (abstract):  "Billing adjustments are primarily driven by plan configuration errors"
```

**Algorithm**: Hierarchical clustering + multi-level summarization.

```python
def abstract_from_instances(specific_facts: list[str], target_level: int) -> str:
    """
    Level 0 → Level 1: Cluster specific instances → summarize each cluster
    Level 1 → Level 2: Cluster patterns → summarize into abstract concept
    """
    # Step 1: Embed all facts
    embeddings = [embed(f) for f in specific_facts]

    # Step 2: Agglomerative clustering (builds hierarchy naturally)
    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.5,
        linkage='average'
    )
    labels = clustering.fit_predict(embeddings)

    # Step 3: For each cluster, LLM-summarize into next level
    abstractions = []
    for cluster_id in set(labels):
        cluster_facts = [f for f, l in zip(specific_facts, labels) if l == cluster_id]
        prompt = f"Summarize these specific observations into one general principle:\n"
        prompt += "\n".join(f"- {f}" for f in cluster_facts)
        abstraction = llm_call(prompt, model="cheap")
        abstractions.append(abstraction)

    return abstractions
```

**Storage**:
- Level 0 (specific) → LFM
- Level 1 (pattern) → SFM
- Level 2 (abstract) → AM as concept nodes

**Trigger**: Sleep mode — when LFM has 10+ related specific facts that haven't been abstracted.

**LLM**: Yes (cheap — summarization).

---

#### 4.7.4 Analogical Learning

**What**: "This is like that" — map structure from a known domain to a new domain. Bootstrap understanding of unfamiliar areas.

**How**:
```
Known domain: Billing
  "billing cycles" → monthly periods → aggregate charges per period

New domain: Collections
  "aging buckets" → ???

Analogy detection:
  embed("billing cycles") ≈ embed("aging buckets") → structural similarity
  Map: aging buckets = time-windowed groupings, like billing cycles
  → apply similar aggregation patterns
```

**Algorithm**: Cross-domain embedding alignment + structure mapping.

```python
def find_analogies(source_domain: str, target_domain: str, threshold: float = 0.6) -> list:
    """Find analogous concepts between two domains in the AM graph."""
    source_entities = kuzu_query(f"MATCH (n:Entity) WHERE n.domain = '{source_domain}' RETURN n")
    target_entities = kuzu_query(f"MATCH (n:Entity) WHERE n.domain = '{target_domain}' RETURN n")

    analogies = []
    for s in source_entities:
        for t in target_entities:
            sim = cosine_similarity(embed(s.description), embed(t.description))
            if sim > threshold and s.name != t.name:
                analogies.append({
                    "source": s.name,
                    "target": t.name,
                    "similarity": sim,
                    "source_domain": source_domain,
                    "target_domain": target_domain
                })

    # Verify with LLM: "Is this a valid analogy?"
    verified = []
    for a in analogies:
        prompt = f"Is '{a['source']}' in {a['source_domain']} analogous to '{a['target']}' in {a['target_domain']}? Explain briefly."
        response = llm_call(prompt, model="expensive")
        if "yes" in response.lower():
            verified.append({**a, "explanation": response})

    return verified
```

**Memory Target**: AM graph — create `is_analogous_to` edges between cross-domain entities.

**Trigger**: Sleep mode — when a new domain is ingested, scan for analogies to existing domains.

**LLM**: Yes (expensive — analogy verification requires reasoning).

---

#### 4.7.5 Corrective Learning

**What**: Learn from explicit corrections. When a user says "that's wrong," update facts, annotate sources, and prevent the same mistake.

**Flow**:
```
Agent says: "ACTV_AMT negative = refund"
User corrects: "No, it can also be a promotional credit"
    ↓
1. Detect correction (NLI: user statement CONTRADICTS agent's last claim)
    ↓
2. Update SFM fact (version increment):
   OLD: "ACTV_AMT: negative = refund"
   NEW: "ACTV_AMT: negative = credit/refund (includes promotional credits)"
    ↓
3. Annotate LFM source with correction note
    ↓
4. Create contrastive pair (see 4.7.10):
   "ACTV_AMT negative ≠ always refund; can be promotional credit"
    ↓
5. Log correction → DuckDB (track error patterns)
    ↓
6. Bandit negative reward for retrieval path that produced wrong answer
    ↓
7. Re-embed updated fact in SFM FAISS
```

**Correction detection**:
```python
def detect_correction(agent_statement: str, user_response: str) -> bool:
    """Use NLI to detect if user is correcting the agent."""
    scores = nli_model.predict([(agent_statement, user_response)])
    # High contradiction + user response contains correction markers
    contradiction_score = scores[0][0]
    correction_markers = ["no", "wrong", "actually", "not correct", "incorrect", "that's not"]
    has_marker = any(m in user_response.lower() for m in correction_markers)
    return contradiction_score > 0.5 and has_marker
```

**Memory Target**: SFM (update fact) + LFM (annotate source) + AM (contrastive edge) + DuckDB (error tracking).

**Trigger**: Real-time — inline during conversation when correction is detected.

**LLM**: Yes (cheap — for rephrasing corrected fact).

---

#### 4.7.6 Transfer Learning

**What**: Apply knowledge from one domain to help in another. Adapt existing procedures and patterns to new contexts.

**How**:
```
Known: Billing domain has a "churn_backtest" procedure (MM)
New:   Collections domain needs a "delinquency_prediction" approach
Transfer: Churn patterns (billing) inform delinquency patterns (collections)
    → Adapt churn_backtest procedure → create collections variant
```

**Algorithm**: Procedure adaptation + cross-domain AM edge creation.

```python
def transfer_procedure(source_proc: str, target_domain: str) -> dict:
    """Adapt a procedure from one domain to another."""
    source_yaml = load_yaml(source_proc)

    prompt = f"""
    This procedure works in the '{source_yaml['domain']}' domain:
    {source_yaml['steps_nl']}

    Adapt it for the '{target_domain}' domain.
    - Map concepts to {target_domain} equivalents
    - Adjust steps for {target_domain} data structures
    - Note what doesn't transfer
    """
    adapted = llm_call(prompt, model="expensive")
    return {
        "name": f"{source_yaml['name']}_{target_domain}_adapted",
        "source": source_proc,
        "domain": target_domain,
        "steps_nl": adapted,
        "source_type": "transfer_learning"
    }
```

**Memory Target**: MM (adapted procedures in `_system_generated/`) + AM (cross-domain edges).

**Trigger**: Sleep mode — when a new domain is added and has few procedures, scan other domains for transferable ones.

**LLM**: Yes (expensive — domain adaptation requires reasoning).

---

#### 4.7.7 Meta-Learning

**What**: Learning HOW to learn — track which learning strategies are most effective and allocate resources accordingly.

**How**: Track every learning event in DuckDB with downstream utility:

```python
# Log every learning event
def log_learning_event(learning_type: str, source: str, output_memory: str, details: dict):
    duckdb_append("learning_events", {
        "ts": now(),
        "learning_type": learning_type,       # "abstraction", "corrective", etc.
        "source": source,                      # what triggered it
        "output_memory": output_memory,        # where result was stored
        "output_id": details.get("id"),
        "llm_cost": details.get("cost", 0),
        "downstream_utility": None             # filled later by utility tracker
    })

# Periodically measure utility: was the learned thing actually used?
def measure_utility():
    """Check if learned memories were accessed and led to good outcomes."""
    recent_learnings = duckdb_query("""
        SELECT * FROM learning_events
        WHERE downstream_utility IS NULL
        AND ts < now() - INTERVAL 7 DAY
    """)
    for event in recent_learnings:
        access_count = get_access_count(event.output_memory, event.output_id)
        sentiment_after = get_avg_sentiment_after(event.ts)
        utility = compute_utility(access_count, sentiment_after)
        duckdb_update(event.id, downstream_utility=utility)
```

**Budget allocation**:
```sql
-- Which learning type has best ROI?
SELECT learning_type,
       AVG(downstream_utility) as avg_utility,
       AVG(llm_cost) as avg_cost,
       AVG(downstream_utility) / NULLIF(AVG(llm_cost), 0) as roi
FROM learning_events
WHERE downstream_utility IS NOT NULL
GROUP BY learning_type
ORDER BY roi DESC
```

Allocate more sleep-mode budget to high-ROI learning types. Reduce budget for learning types that produce low-utility knowledge.

**Memory Target**: DuckDB (learning effectiveness metrics) + MM (learning strategy configurations).

**Trigger**: Sleep mode — weekly review of learning effectiveness.

**LLM**: No (pure analytics).

---

#### 4.7.8 Incremental / Online Learning

**What**: Continuously update models and indexes without full retraining. The agent gets smarter with every interaction without a "retrain" step.

**What updates incrementally**:

| Component | How | When |
|---|---|---|
| FAISS indexes | `index.add()` for new vectors. No full rebuild needed. | On every new memory entry |
| LinUCB bandit weights | `update(arm, features, reward)` — matrix update, O(d²) | After every retrieval + outcome |
| BM25 index | Add new documents to corpus, update IDF | On new SFM/LFM entries |
| Ebbinghaus retention scores | Recalculate on access | On every memory access |
| Time-decay scores | Recalculate on query | On every retrieval ranking |
| Access counts | `+= 1` | On every retrieval hit |

**Key principle**: No batch retraining needed for any component. Everything updates online, one observation at a time.

**FAISS periodic rebuild**: While incremental `add()` is fine for growth, periodic rebuild (sleep mode) compacts the index, removes deleted vectors, and re-clusters for IVF indexes.

**Memory Target**: All FAISS indexes, bandit weights, scoring parameters.

**Trigger**: Real-time (incremental updates) + sleep mode (periodic compaction).

**LLM**: No (pure computation).

---

#### 4.7.9 Observational Learning

**What**: Learn from watching admin/expert behavior. When an admin manually performs a task, the agent observes the action trace and internalizes it as a procedure.

**How**:
```
Admin manually runs a sequence:
  1. Opens billing data
  2. Filters ACTV_CODE = 'ADJ'
  3. Groups by SOC
  4. Computes sum of ACTV_AMT
  5. Exports result
    ↓
Agent captures action trace:
  [open_dataset("comp_plan_disc"), filter("ACTV_CODE='ADJ'"),
   group_by("SOC"), aggregate("SUM(ACTV_AMT)"), export()]
    ↓
LLM: "Convert this action trace into a reusable procedure"
    ↓
Output: adjustment_analysis.py + adjustment_analysis.yaml
    ↓
Save to MM/_system_generated/ → register in Meta
```

**Implementation**:
```python
class ActionObserver:
    def __init__(self):
        self.traces: dict[str, list] = {}  # session_id → action list

    def record(self, session_id: str, action: str, params: dict):
        """Hook into admin tool calls to record actions."""
        if session_id not in self.traces:
            self.traces[session_id] = []
        self.traces[session_id].append({
            "action": action,
            "params": params,
            "timestamp": now()
        })

    def extract_procedure(self, session_id: str) -> dict:
        """Convert observed action trace to a procedure."""
        trace = self.traces[session_id]
        prompt = f"""
        An expert performed these steps:
        {json.dumps(trace, indent=2)}

        Convert this into a reusable procedure with:
        - A descriptive name
        - Parameterized steps (replace specific values with parameters)
        - Natural language description of each step
        """
        return llm_call(prompt, model="cheap")
```

**Memory Target**: MM (new procedures in `_system_generated/`).

**Trigger**: End of admin session — if action trace length > threshold (e.g., 3+ steps).

**LLM**: Yes (cheap — procedure generation from trace).

---

#### 4.7.10 Contrastive Learning

**What**: Learn what something IS by understanding what it ISN'T. Store positive-negative fact pairs to reduce hallucination and confusion.

**Fact pairs**:
```
✅ "ACTV_AMT is the monetary amount of a billing event"
❌ "ACTV_AMT is NOT the payment received amount"
❌ "ACTV_AMT is NOT the account balance"

✅ "SUB_STATUS = 'C' means cancelled"
❌ "SUB_STATUS = 'C' does NOT mean 'completed' or 'closed'"
```

**Sources of contrastive pairs**:
1. **Corrective learning** (4.7.5): Every user correction generates a contrastive pair
2. **Confusion detection**: When the agent retrieves fact A but the user wanted fact B → A and B are "commonly confused"
3. **Admin-curated**: Domain experts define "common mistakes" upfront
4. **Sleep mode generation**: LLM reviews SFM facts and generates "what this is NOT" counterparts

**AM graph representation**:
```cypher
CREATE REL TABLE NOT_SAME_AS (FROM Entity TO Entity, explanation STRING)
CREATE REL TABLE COMMONLY_CONFUSED_WITH (FROM Entity TO Entity, confusion_count INT64)
```

**Recall integration**: When retrieving a fact, also fetch `NOT_SAME_AS` edges → include in LLM context to prevent hallucination:

```python
def retrieve_with_contrast(entity_id: str) -> dict:
    fact = get_fact(entity_id)
    contrasts = kuzu_query(f"""
        MATCH (a:Entity)-[r:NOT_SAME_AS]->(b:Entity)
        WHERE a.id = '{entity_id}'
        RETURN r.explanation, b.name
    """)
    return {
        "fact": fact,
        "not_this": [c["explanation"] for c in contrasts]
    }
```

**Memory Target**: SFM (paired facts) + AM (`NOT_SAME_AS` and `COMMONLY_CONFUSED_WITH` edges).

**Trigger**: Real-time (from corrections) + sleep mode (LLM-generated negatives).

**LLM**: Yes (cheap — generating "what it's not" counterparts).

---

### 4.8 Learning Taxonomy — Complete Summary

| # | Learning Type | Algorithm / Approach | LLM? | Cost | Memory Target | Trigger |
|---|---|---|---|---|---|---|
| 1 | Reinforced | LinUCB contextual bandit | No | Free | Bandit weights | Real-time |
| 2 | Generalization | DBSCAN + LCS + distillation | Yes | Expensive | MM procedures | Sleep |
| 3 | Abstraction | Hierarchical clustering + multi-level summarization | Yes | Cheap | SFM + AM | Sleep |
| 4 | Analogical | Cross-domain embedding alignment + LLM verification | Yes | Expensive | AM edges | Sleep |
| 5 | Corrective | NLI correction detection + fact versioning | Yes | Cheap | SFM + LFM + AM | Real-time |
| 6 | Transfer | Procedure adaptation + cross-domain edges | Yes | Expensive | MM + AM | Sleep |
| 7 | Meta-Learning | Learning effectiveness tracking + budget optimization | No | Free | DuckDB + MM | Sleep (weekly) |
| 8 | Incremental | Online index updates + streaming bandit updates | No | Free | FAISS + bandit | Real-time |
| 9 | Observational | Action trace capture + procedure extraction | Yes | Cheap | MM procedures | End of admin session |
| 10 | Contrastive | Positive-negative pairing + NOT_SAME_AS edges | Yes | Cheap | SFM + AM | Real-time + Sleep |

**Cost breakdown**: 4 free (no LLM) + 3 cheap + 3 expensive. Sleep-mode budget cap (section 4.3) manages the expensive ones.

**Learning cycle**:
```
Real-time (during conversation):
  → Reinforced (bandit update)
  → Corrective (fix mistakes immediately)
  → Incremental (update indexes)
  → Contrastive (log confusion pairs)

Sleep mode (off-peak):
  → Generalization (find patterns → procedures)
  → Abstraction (specific → general)
  → Analogical (cross-domain mapping)
  → Transfer (adapt procedures)
  → Contrastive (generate negatives)

Weekly review:
  → Meta-learning (which learning types are worth the cost?)
  → Adjust budget allocation
```

---

## 5. Response Layer

A separate module from Memory. This is the agent's processing pipeline — receives user input, routes it, reasons over it, executes skills, and produces a response.

---

### 5.1 High-Level Flow

```
User question
    ↓
Store to Working Memory (WM)
    ↓
┌─────────────────────────────────────────────┐
│              GATE (SLM / Rule-based)         │
│  Is this a greeting, chitchat, or trivia?    │
├──────────────┬──────────────────────────────┤
│  YES         │  NO (everything else)         │
│  ↓           │  ↓                            │
│  Respond     │  Forward to Deep Pipeline     │
│  directly    │                               │
└──────────────┴──────────────────────────────┘
    ↓                              ↓
Store to WM → Return          Store to WM → Return
```

The Gate is a **binary classifier**, not a "try and see." Greetings/chitchat/trivia → handle inline. Everything else → Deep Pipeline. No confidence scoring needed.

---

### 5.2 Gate (`response/gate.py`)

**Purpose**: Fast, cheap filter for trivial interactions. Prevents expensive Deep Pipeline calls for "good morning" or "what's the capital of Texas."

**Implementation options** (in order of simplicity):

| Option | How | When to Use |
|---|---|---|
| **Rule-based** | Regex + keyword matching + intent patterns | Start here |
| **SLM** | Small language model (e.g., DistilBERT intent classifier) | When rule-based gets brittle |
| **Tiny LLM** | GPT-4o-mini with a simple system prompt | If SLM accuracy is insufficient |

**Rule-based Gate** (starting implementation):
```python
class Gate:
    GREETING_PATTERNS = ["hello", "hi", "good morning", "hey", "howdy", ...]
    CHITCHAT_PATTERNS = ["how are you", "what's up", "thank you", ...]

    def classify(self, message: str) -> str:
        """Returns 'gate' (handle here) or 'deep' (forward to pipeline)."""
        lower = message.lower().strip()

        # Greeting / chitchat
        if any(p in lower for p in self.GREETING_PATTERNS + self.CHITCHAT_PATTERNS):
            return "gate"

        # General knowledge (no domain terms detected)
        if not self._has_domain_terms(lower) and self._is_simple_question(lower):
            return "gate"

        return "deep"

    def respond(self, message: str) -> str:
        """Generate response for gate-level messages (no LLM needed)."""
        # Template-based or SLM response
        ...
```

**User rejection escalation**: If the user says "that's not what I meant" or "I need more detail" after a Gate response → auto-escalate to Deep Pipeline with the original question + rejection context.

---

### 5.3 Deep Pipeline (`response/pipeline.py`)

**Purpose**: Full reasoning pipeline for all non-trivial questions. Orchestrates 6 subsystems to produce high-quality responses.

**Architecture**:

```
Complex query arrives
    ↓
┌──────────────────────────────────────────────────────────┐
│                      DEEP PIPELINE                        │
│                                                           │
│  ① THINKING       → Understand the problem, plan steps   │
│       ↓                                                   │
│  ② MEMORY RECALL  → Retrieve context from Memory Layer   │
│       ↓               (calls MML recall engine)           │
│  ③ DECISION       → What approach? Which skills needed?   │
│       ↓               What can be parallelized?           │
│  ④ SKILL EXEC     → Run scripts, query DBs, call APIs    │
│       ↓               Spawn sub-agents if parallel work   │
│  ⑤ CREATIVITY     → Generate insights, alternatives,     │
│       ↓               novel combinations, visualizations  │
│  ⑥ PREDICTION     → Forecast outcomes, risk assessment,   │
│       ↓               confidence estimation                │
│                                                           │
│  SYNTHESIS LLM    → Combine all outputs into final        │
│                      response                              │
└──────────────────────────────────────────────────────────┘
    ↓
Store to WM → Return
```

**Important**: Not always sequential. The Deep LLM orchestrates and decides which subsystems to invoke. Some queries need only ① + ② . Others need all 6.

---

#### 5.3.1 Thinking (`response/thinking.py`)

**What**: Break down the problem, identify what's being asked, plan the approach.

**How**: The Deep LLM's chain-of-thought reasoning. Produces a structured plan:

```python
@dataclass
class ThoughtPlan:
    understanding: str          # What is the user actually asking?
    complexity: str             # "simple_lookup" | "analysis" | "multi_step" | "creative"
    steps: list[str]            # Planned steps to solve
    memory_needed: list[str]    # Which memory types to query
    skills_needed: list[str]    # Which skills to invoke
    parallelizable: list[list]  # Groups of steps that can run in parallel
    confidence: float           # How confident the plan is (0-1)
```

---

#### 5.3.2 Memory Recall (`response/recall_bridge.py`)

**What**: Bridge between Response Layer and Memory Layer. Forwards the query to MML's recall engine and returns enriched context.

**How**: Calls the Memory Management Layer's recall engine (Section 4.1). The Response Layer does NOT access memory types directly — always goes through MML.

```python
class RecallBridge:
    def __init__(self, mml: MemoryManagementLayer):
        self.mml = mml

    async def recall(self, query: str, plan: ThoughtPlan) -> RecallResult:
        """Retrieve relevant context from all memory types via MML."""
        # MML handles tiered search: Meta → SFM → LFM → AM
        context = await self.mml.recall(
            query=query,
            memory_hints=plan.memory_needed,  # from thinking step
            max_results=10
        )
        return context  # ranked, deduplicated, with confidence scores
```

---

#### 5.3.3 Decision Making (`response/decision.py`)

**What**: Based on the thought plan and recalled context, decide the execution strategy.

**Decisions made**:
- Which skills to execute and in what order
- Whether to spawn sub-agents for parallel work
- Whether on-the-fly code generation is needed
- Whether creativity or prediction subsystems are needed
- Resource allocation (which LLM tier for each step)

```python
@dataclass
class ExecutionPlan:
    strategy: str               # "direct_answer" | "skill_execution" | "multi_agent" | "code_gen"
    skill_calls: list[SkillCall]    # Ordered list of skill invocations
    parallel_groups: list[list[SkillCall]]  # Skills that can run concurrently
    needs_code_gen: bool        # Deep LLM should generate code
    needs_creativity: bool      # Invoke creativity subsystem
    needs_prediction: bool      # Invoke prediction subsystem
    sub_agents: list[SubAgentSpec]  # Sub-agents to spawn
```

---

#### 5.3.4 Skill Execution (`response/skills/`)

**What**: The runtime infrastructure that actually executes logic — Python scripts, database queries, API calls. The "logic to execute the logic."

**Skill Sources**:
1. **Motor Memory (MM) procedures** — Pre-built `.py` + `.yaml` from `memory/procedures/`
2. **On-the-fly code generation** — Deep LLM generates Python for novel problems

**Skills Execution Engine**:

```
┌─────────────────────────────────────────────────────────┐
│                Skills Execution Engine                    │
│                                                          │
│  ┌──────────────────┐   ┌──────────────────┐            │
│  │  Script Runner    │   │  DB Connector     │           │
│  │                   │   │                   │           │
│  │  - Sandboxed exec │   │  - Snowflake      │           │
│  │  - Timeout guard  │   │  - SQLite         │           │
│  │  - stdout/stderr  │   │  - DuckDB         │           │
│  │    capture        │   │  - Auth management │           │
│  │  - Resource limits│   │  - Connection pool │           │
│  └──────────────────┘   └──────────────────┘            │
│                                                          │
│  ┌──────────────────┐   ┌──────────────────┐            │
│  │  API Client       │   │  Code Generator   │           │
│  │                   │   │                   │           │
│  │  - Azure SDK      │   │  - Deep LLM       │           │
│  │  - REST calls     │   │    generates code  │           │
│  │  - Retry logic    │   │  - Syntax validated│           │
│  │  - Auth rotation  │   │  - Import whitelist│           │
│  │  - Rate limiting  │   │  - Sandboxed exec  │           │
│  └──────────────────┘   │  - One-shot (not   │           │
│                          │    auto-saved to MM)│           │
│  ┌──────────────────────┴──────────────────┐            │
│  │  Sub-Agent Spawner                       │            │
│  │  - asyncio.gather for parallel execution │            │
│  │  - Each sub-agent has memory access      │            │
│  │  - Results aggregated by Deep LLM        │            │
│  │  - Timeout + cancellation support        │            │
│  └──────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────┘
```

**Skill Definition**:

```python
@dataclass
class Skill:
    name: str                    # "query_billing_data"
    description: str             # For LLM to understand when to use it
    parameters: dict             # JSON schema of expected inputs
    executor: Callable           # The actual Python function
    skill_type: str              # "python" | "sql" | "api" | "composite"
    timeout_seconds: int         # Max execution time
    requires_approval: bool      # Human-in-the-loop for dangerous ops
    source: str                  # "motor_memory" | "generated"
```

**Skill Types**:

| Type | What | Runtime | Example |
|---|---|---|---|
| `python` | Execute a Python function | Script Runner (sandboxed) | Churn analysis, data transformation |
| `sql` | Query a database | DB Connector (auth + pool) | Retrieve billing records from Snowflake |
| `api` | Call an external service | API Client (retry + auth) | Azure Blob operations, internal APIs |
| `composite` | Orchestrate multiple skills | Sub-Agent Spawner | "Pull data → transform → analyze → report" |

**DB Connector — Auth & Connection Management**:

```python
class DBConnector:
    def __init__(self, config_path: str):
        self.pools: dict[str, ConnectionPool] = {}
        self.config = load_config(config_path)  # encrypted credentials

    def get_connection(self, db_name: str) -> Connection:
        """Get a pooled connection with auth handled."""
        if db_name not in self.pools:
            self.pools[db_name] = self._create_pool(db_name)
        return self.pools[db_name].acquire()

    def execute_query(self, db_name: str, query: str, params: dict = None) -> list[dict]:
        """Execute SQL with parameterized queries (prevent injection)."""
        conn = self.get_connection(db_name)
        try:
            return conn.execute(query, params).fetchall()
        finally:
            conn.release()
```

**On-the-fly Code Generation — Security**:

```python
ALLOWED_IMPORTS = {
    "pandas", "numpy", "json", "datetime", "math",
    "collections", "itertools", "functools", "re"
}
BLOCKED_MODULES = {
    "os", "sys", "subprocess", "shutil", "socket",
    "http", "urllib", "importlib", "__builtins__"
}

class CodeGenerator:
    def generate_and_execute(self, task: str, context: dict) -> Any:
        # 1. Deep LLM generates code
        code = self.deep_llm.generate_code(task, context)

        # 2. Validate
        self._check_syntax(code)
        self._check_imports(code, ALLOWED_IMPORTS, BLOCKED_MODULES)

        # 3. Execute in sandbox
        result = self._sandboxed_exec(code, timeout=30, context=context)

        # 4. NOT auto-saved to MM — one-shot unless admin promotes
        return result
```

---

#### 5.3.5 Creativity (`response/creativity.py`)

**What**: Generate novel insights, alternative perspectives, unexpected connections, and visualizations that go beyond direct retrieval.

**When invoked**: Decision engine determines the query needs creative output — "what if" scenarios, brainstorming, data storytelling, alternative interpretations.

```python
@dataclass
class CreativeOutput:
    insights: list[str]         # Novel observations from the data
    alternatives: list[str]     # Alternative interpretations / approaches
    connections: list[str]      # Unexpected links (leverages AM graph)
    visualizations: list[dict]  # Suggested charts/graphs with specs
```

**How**: Deep LLM with a creativity-focused prompt + AM graph traversal for finding non-obvious connections between entities.

---

#### 5.3.6 Prediction (`response/prediction.py`)

**What**: Forecast outcomes, estimate risks, project trends based on available data and context.

**When invoked**: Queries involving "what will happen," "forecast," "predict," "estimate," trend analysis.

```python
@dataclass
class PredictionOutput:
    prediction: str             # The forecast / estimate
    confidence: float           # 0.0 - 1.0
    reasoning: str              # Why this prediction
    assumptions: list[str]      # What assumptions were made
    risks: list[str]            # What could make this wrong
    data_basis: list[str]       # What data supports this
```

**How**: Deep LLM reasoning over recalled data + skill execution results. For numerical predictions, can generate and execute statistical models on-the-fly (via Code Generator).

---

### 5.4 Sub-Agent System (`response/sub_agents.py`)

**Purpose**: Spawn parallel workers for complex multi-step tasks.

```python
@dataclass
class SubAgentSpec:
    task: str                   # What this sub-agent should do
    skills: list[str]           # Skills it's authorized to use
    memory_access: bool         # Can it query memory? (yes, via MML)
    timeout: int                # Max seconds
    priority: int               # Execution priority

class SubAgentSpawner:
    async def execute_parallel(self, specs: list[SubAgentSpec]) -> list[SubAgentResult]:
        """Spawn sub-agents as parallel asyncio tasks."""
        tasks = [self._run_sub_agent(spec) for spec in specs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [
            SubAgentResult(spec=s, result=r, success=not isinstance(r, Exception))
            for s, r in zip(specs, results)
        ]

    async def _run_sub_agent(self, spec: SubAgentSpec) -> Any:
        """Single sub-agent: recall → plan → execute → return result."""
        context = await self.mml.recall(spec.task) if spec.memory_access else {}
        result = await self.skill_engine.execute(spec.skills, spec.task, context)
        return result
```

**Example**:
```
User: "Compare March vs April revenue leakage across all markets"
    ↓
Deep LLM → Decision: spawn 2 sub-agents
    ↓
Sub-agent A: "Get March revenue leakage" → skill: query_billing(month="2026-03")
Sub-agent B: "Get April revenue leakage" → skill: query_billing(month="2026-04")
    ↓ (asyncio.gather — parallel)
    ↓ both complete
Deep LLM: Synthesize comparison → format response → return
```

---

### 5.5 Response Synthesis (`response/synthesis.py`)

**What**: Final step — combine all outputs (recall, skill results, creativity, prediction) into a coherent response.

```python
class ResponseSynthesizer:
    async def synthesize(
        self,
        query: str,
        plan: ThoughtPlan,
        recall_result: RecallResult,
        skill_results: list[SkillResult],
        creative_output: CreativeOutput | None,
        prediction_output: PredictionOutput | None,
        sub_agent_results: list[SubAgentResult]
    ) -> str:
        """Deep LLM synthesizes everything into final response."""
        context = self._build_context(
            recall_result, skill_results, creative_output,
            prediction_output, sub_agent_results
        )
        response = await self.deep_llm.generate(
            system=self._get_system_prompt(plan),
            context=context,
            query=query
        )
        return response
```

---

### 5.6 Complete Flow — Putting It All Together

```
User message arrives
    ↓
WM: Store message
    ↓
Gate: greeting/chitchat/trivia?
    ↓ YES → respond directly → store to WM → return
    ↓ NO
    ↓
Deep Pipeline:
    ↓
① Thinking: understand query → produce ThoughtPlan
    ↓
② Recall Bridge: query MML → get relevant context
    ↓
③ Decision: produce ExecutionPlan (skills, sub-agents, code gen?)
    ↓
④ Skill Execution:
    ├─ MM procedures → Script Runner / DB Connector / API Client
    ├─ Code Generator → sandbox exec (if novel problem)
    └─ Sub-Agent Spawner → parallel tasks (if needed)
    ↓
⑤ Creativity (if needed): insights, alternatives, connections
    ↓
⑥ Prediction (if needed): forecasts, confidence, risks
    ↓
Synthesis LLM: combine all outputs → final response
    ↓
WM: Store response
    ↓
EM: Capture sentiment signal
    ↓
Analytics: Log metrics (latency, skills used, LLM calls)
    ↓
Return to user
```

---

### 5.7 Design Decisions

| Decision | Rationale |
|---|---|
| Gate as binary classifier, not confidence-based | Simple, fast, no gray area. Chitchat vs everything else. |
| Gate is SLM/rule-based, not LLM | Zero cost for trivial messages. LLM is overkill for "hello." |
| Response Layer separate from Memory | Single responsibility. Response Layer consumes memory via MML bridge, doesn't manage it. |
| Deep Pipeline subsystems are LLM-orchestrated | Not every query needs all 6 subsystems. LLM decides what to invoke based on ThoughtPlan. |
| Skills from MM + on-the-fly code gen | Pre-built handles known patterns. Code gen handles novel problems. Best of both. |
| On-the-fly code NOT auto-saved | Security + quality gate. Admin must review and promote to MM. |
| Sandboxed execution | Generated code runs in restricted environment. Whitelist imports, no system access. |
| DB auth managed centrally | Connection pooling, credential rotation, parameterized queries — all in one place. Prevents injection. |
| Sub-agents via asyncio | Lightweight parallel execution. No separate processes or containers needed. |

---

### 5.8 Codebase Structure

```
response/                            ← Response Layer (separate from memory/)
├── __init__.py
├── gate.py                          ← Gate classifier (SLM / rule-based)
├── pipeline.py                      ← Deep Pipeline orchestrator
├── thinking.py                      ← Problem decomposition + planning
├── recall_bridge.py                 ← Bridge to Memory Layer (MML)
├── decision.py                      ← Execution strategy selection
├── creativity.py                    ← Novel insights + alternatives
├── prediction.py                    ← Forecasting + risk assessment
├── synthesis.py                     ← Final response assembly
├── sub_agents.py                    ← Parallel sub-agent spawner
└── skills/                          ← Skills Execution Engine
    ├── __init__.py
    ├── engine.py                    ← Skill registry + dispatch
    ├── script_runner.py             ← Sandboxed Python execution
    ├── db_connector.py              ← DB auth, pooling, query execution
    ├── api_client.py                ← Azure SDK, REST, retry, auth
    └── code_generator.py            ← On-the-fly code gen + validation
```

---

## 6. Control Systems

Three brain-inspired control networks that coordinate the entire agent. Without these, Memory, Response, Analytics, and Scheduling would compete for resources blindly. These are the "brain" that makes all the layers work as one system.

---

### 6.1 Salience Network (`control/salience.py`)

**Purpose**: The attention/priority arbiter — decides what deserves the agent's resources RIGHT NOW. Always on, lightweight, event-driven. Analogous to the human salience network that detects what's important and triggers mode switches.

**Key Properties**:
- **Always on** — Listens for events from all systems continuously.
- **Lightweight** — Rule-based scoring + simple math. No LLM calls.
- **Event-driven** — Reacts to signals, doesn't poll.
- **Outputs priority decisions** — Tells CEN what to focus on and in what order.

**Input Events**:

| Source | Event | Example |
|---|---|---|
| User | New message | "What's March revenue leakage?" |
| PM Scheduler | Task due | Cron job fires: daily billing report |
| MML | Conflict detected | SFM fact contradicts LFM chunk |
| MML | Memory health alert | FAISS index corruption detected |
| Sub-agents | Task complete | Parallel query finished |
| System | Resource pressure | Redis memory > 80%, LLM budget low |
| EM | Sentiment shift | User went from neutral → frustrated |
| Analytics | Anomaly | Unusual spike in error rate |

**Priority Scoring**:

$$P(event) = w_{type} \cdot T + w_{urgency} \cdot U + w_{impact} \cdot I + w_{decay} \cdot D(t)$$

Where:
- $T$ = type priority (user interaction > scheduled task > maintenance)
- $U$ = urgency (due now > due in 5 min > due tonight)
- $I$ = impact (affects user response > affects system health > affects optimization)
- $D(t)$ = time decay (events waiting longer get boosted to prevent starvation)

```python
class SalienceNetwork:
    # Priority tiers (higher = more important)
    TYPE_WEIGHTS = {
        "user_message": 100,
        "user_frustrated": 95,
        "scheduled_task_due": 70,
        "sub_agent_complete": 60,
        "conflict_detected": 50,
        "health_alert": 80,
        "resource_pressure": 75,
        "anomaly": 40,
        "maintenance": 10,
    }

    def __init__(self):
        self.event_queue = asyncio.PriorityQueue()
        self.listeners: list[Callable] = []

    async def ingest_event(self, event: AgentEvent):
        """Score and queue an incoming event."""
        priority = self._score(event)
        await self.event_queue.put((-priority, event))  # negative for max-priority-first
        # Notify CEN immediately for high-priority events
        if priority >= 80:
            await self._notify_cen(event, priority)

    def _score(self, event: AgentEvent) -> float:
        T = self.TYPE_WEIGHTS.get(event.type, 10)
        U = self._urgency_score(event)
        I = self._impact_score(event)
        D = self._decay_boost(event.created_at)
        return 0.4 * T + 0.25 * U + 0.25 * I + 0.1 * D

    def _urgency_score(self, event: AgentEvent) -> float:
        """How time-sensitive is this?"""
        if event.deadline is None:
            return 50  # no deadline = medium
        seconds_until = (event.deadline - now()).total_seconds()
        if seconds_until <= 0: return 100    # overdue
        if seconds_until <= 60: return 90    # due within 1 min
        if seconds_until <= 300: return 70   # due within 5 min
        if seconds_until <= 3600: return 40  # due within 1 hour
        return 20

    def _decay_boost(self, created_at: datetime) -> float:
        """Older waiting events get priority boost to prevent starvation."""
        wait_seconds = (now() - created_at).total_seconds()
        return min(100, wait_seconds / 60 * 10)  # +10 per minute waiting, cap at 100

    async def get_next(self) -> AgentEvent:
        """CEN calls this to get the next highest-priority event."""
        _, event = await self.event_queue.get()
        return event
```

**Mode Detection** — switches between active and idle:

```python
class ModeDetector:
    IDLE_THRESHOLD_SECONDS = 300  # 5 min no user interaction → idle

    def __init__(self):
        self.last_user_interaction = now()
        self.current_mode = "active"  # "active" | "idle"

    def on_user_message(self):
        self.last_user_interaction = now()
        if self.current_mode == "idle":
            self.current_mode = "active"
            return "switch_to_active"  # signal CEN to wake up

    def check_idle(self) -> str | None:
        if self.current_mode == "active":
            idle_time = (now() - self.last_user_interaction).total_seconds()
            if idle_time > self.IDLE_THRESHOLD_SECONDS:
                self.current_mode = "idle"
                return "switch_to_idle"  # signal CEN to enter DMN mode
        return None
```

---

### 6.2 Central Executive Network (`control/cen.py`)

**Purpose**: The global orchestrator — the `main()` of the agent. Coordinates Memory, Response, Analytics, and Scheduling as one system. Consumes priority signals from Salience Network, dispatches work, manages resources.

**Key Properties**:
- **The event loop** — All work flows through CEN. It's the single coordination point.
- **Resource-aware** — Tracks concurrent requests, LLM budget, memory pressure, active sub-agents.
- **Mode-switching** — Switches between active mode (serving users) and idle mode (DMN takes over).
- **Non-blocking** — Uses asyncio to handle multiple concerns concurrently.

**Architecture**:

```
              Salience Network
                    ↓ (priority events)
┌──────────────────────────────────────────────────────────┐
│                CENTRAL EXECUTIVE NETWORK                  │
│                                                           │
│  ┌─────────────────────────────────────────────────┐     │
│  │              Event Loop (asyncio)                │     │
│  │                                                  │     │
│  │  Priority event arrives                          │     │
│  │      ↓                                           │     │
│  │  Resource check: can we handle this now?         │     │
│  │      ↓ yes              ↓ no                     │     │
│  │  Dispatch to:           Queue / throttle         │     │
│  │  - Response Layer                                │     │
│  │  - PM Scheduler                                  │     │
│  │  - MML (maintenance)                             │     │
│  │  - DMN (idle tasks)                              │     │
│  │      ↓                                           │     │
│  │  Monitor execution                               │     │
│  │      ↓                                           │     │
│  │  Collect result → route to next step             │     │
│  └─────────────────────────────────────────────────┘     │
│                                                           │
│  ┌─────────────────────────────────────────────────┐     │
│  │           Resource Manager                       │     │
│  │                                                  │     │
│  │  - concurrent_requests: int (max N)              │     │
│  │  - active_sub_agents: int (max M)                │     │
│  │  - llm_budget_remaining: float                   │     │
│  │  - memory_pressure: float (0-1)                  │     │
│  │  - cpu_load: float (0-1)                         │     │
│  └─────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

**Implementation**:

```python
class CentralExecutiveNetwork:
    def __init__(
        self,
        salience: SalienceNetwork,
        response_layer: ResponseLayer,
        mml: MemoryManagementLayer,
        pm_scheduler: ProspectiveMemoryScheduler,
        dmn: DefaultModeNetwork,
        analytics: AnalyticsLayer,
    ):
        self.salience = salience
        self.response = response_layer
        self.mml = mml
        self.pm = pm_scheduler
        self.dmn = dmn
        self.analytics = analytics
        self.resources = ResourceManager()
        self.mode = "active"

    async def run(self):
        """Main event loop — the brain of the agent."""
        # Start background listeners
        asyncio.create_task(self._mode_monitor())
        asyncio.create_task(self._pm_listener())

        while True:
            event = await self.salience.get_next()
            await self._handle_event(event)

    async def _handle_event(self, event: AgentEvent):
        """Route event to the appropriate system."""
        # Log to analytics
        self.analytics.log_metric("cen", "event_received", 1, {"type": event.type})

        match event.type:
            # User-facing events — highest priority
            case "user_message":
                if self.resources.can_accept_request():
                    self.resources.acquire_request()
                    try:
                        response = await self.response.handle(event.payload)
                        await self._deliver_response(event, response)
                    finally:
                        self.resources.release_request()
                else:
                    await self._queue_with_backpressure(event)

            # Scheduled tasks
            case "scheduled_task_due":
                if self.resources.can_run_task():
                    await self.pm.execute_task(event.payload)
                else:
                    await self._defer_task(event, delay_seconds=30)

            # Sub-agent completion
            case "sub_agent_complete":
                await self.response.handle_sub_agent_result(event.payload)
                self.resources.release_sub_agent()

            # Memory health
            case "conflict_detected":
                self.mml.queue_conflict(event.payload)

            case "health_alert":
                await self._handle_health_alert(event)

            # Sentiment escalation
            case "user_frustrated":
                await self._escalate_response(event)

            # System resource pressure
            case "resource_pressure":
                await self._apply_backpressure(event)

    async def _mode_monitor(self):
        """Monitor for mode switches between active and idle."""
        while True:
            await asyncio.sleep(60)  # check every minute
            mode_signal = self.salience.mode_detector.check_idle()
            if mode_signal == "switch_to_idle" and self.mode == "active":
                self.mode = "idle"
                asyncio.create_task(self.dmn.activate())
                self.analytics.log_metric("cen", "mode_switch", 1, {"to": "idle"})
            elif mode_signal == "switch_to_active" and self.mode == "idle":
                self.mode = "active"
                self.dmn.deactivate()
                self.analytics.log_metric("cen", "mode_switch", 1, {"to": "active"})

    async def _escalate_response(self, event: AgentEvent):
        """User is frustrated — reallocate resources to their request."""
        # Pause low-priority background tasks
        self.mml.pause_background(priority_below="P1")
        # Re-process with more context and higher-quality model
        await self.response.handle(event.payload, escalated=True)

    async def _apply_backpressure(self, event: AgentEvent):
        """System under pressure — reduce load."""
        if event.payload["resource"] == "llm_budget":
            self.response.switch_to_cheaper_model()
        elif event.payload["resource"] == "memory":
            self.mml.trigger_emergency_cleanup()
        elif event.payload["resource"] == "cpu":
            self.resources.reduce_max_concurrent(by=2)


class ResourceManager:
    def __init__(self, max_concurrent_requests=5, max_sub_agents=10):
        self.max_requests = max_concurrent_requests
        self.max_sub_agents = max_sub_agents
        self.active_requests = 0
        self.active_sub_agents = 0
        self.llm_budget = BudgetManager()

    def can_accept_request(self) -> bool:
        return self.active_requests < self.max_requests

    def can_run_task(self) -> bool:
        return self.active_requests < self.max_requests - 1  # reserve 1 slot for users

    def can_spawn_sub_agent(self) -> bool:
        return self.active_sub_agents < self.max_sub_agents

    def acquire_request(self): self.active_requests += 1
    def release_request(self): self.active_requests -= 1
    def acquire_sub_agent(self): self.active_sub_agents += 1
    def release_sub_agent(self): self.active_sub_agents -= 1
```

**Resource Decisions**:

| Situation | CEN Action |
|---|---|
| User message + full capacity | Queue with backpressure, respond with "processing" |
| User frustrated | Pause background tasks, escalate response with better model |
| LLM budget low | Switch response layer to cheaper model |
| Scheduled task + user active | Reserve 1 slot for users, run task only if capacity allows |
| Memory pressure (Redis > 80%) | Trigger emergency WM summarization |
| CPU load high | Reduce max concurrent requests |

---

### 6.3 Default Mode Network (`control/dmn.py`)

**Purpose**: What the agent does when no one is asking — self-reflection, spontaneous planning, and proactive maintenance. Activated by CEN when Salience Network detects idle mode. Enhanced version of MML sleep mode.

**Key Properties**:
- **Activated on idle** — CEN switches to DMN when no user interaction for 5+ minutes.
- **Interruptible** — Instantly yields to CEN when a user message arrives.
- **Self-aware** — Reviews own performance, identifies weak areas, plans improvements.
- **Proactive** — Anticipates user needs based on patterns.

**Three modes of operation**:

```
DMN Activated (agent is idle)
    ↓
┌─────────────────────────────────────────┐
│  Phase 1: Self-Reflection               │
│  "How did I perform recently?"           │
│                                          │
│  Phase 2: Spontaneous Planning           │
│  "What should I prepare for?"            │
│                                          │
│  Phase 3: Maintenance                    │
│  (MML sleep tasks — consolidation, etc.) │
└─────────────────────────────────────────┘
    ↓ (interrupted by user message)
    CEN switches back to active mode
```

---

#### 6.3.1 Self-Reflection (`control/dmn.py` → `reflect()`)

**What**: Agent reviews its own recent performance. Produces actionable self-assessment.

```python
class DefaultModeNetwork:
    def __init__(self, mml, analytics, pm):
        self.mml = mml
        self.analytics = analytics
        self.pm = pm
        self.active = False
        self._cancel_event = asyncio.Event()

    async def activate(self):
        """Start DMN processing. Runs until deactivated."""
        self.active = True
        self._cancel_event.clear()

        try:
            # Phase 1: Self-reflection
            await self._reflect()
            if not self.active: return

            # Phase 2: Spontaneous planning
            await self._plan()
            if not self.active: return

            # Phase 3: Maintenance (delegate to MML)
            await self.mml.run_sleep_tasks(cancel_event=self._cancel_event)
        except asyncio.CancelledError:
            pass

    def deactivate(self):
        """CEN calls this when switching back to active mode."""
        self.active = False
        self._cancel_event.set()

    async def _reflect(self):
        """Self-reflection: How did I perform?"""
        # Query analytics for recent performance
        report = self.analytics.query("""
            SELECT
                metric_name,
                AVG(metric_value) as avg_val,
                MIN(metric_value) as min_val,
                MAX(metric_value) as max_val,
                COUNT(*) as count
            FROM metrics
            WHERE ts > now() - INTERVAL 24 HOUR
            AND component IN ('response', 'recall', 'skills')
            GROUP BY metric_name
        """)

        # Identify weak areas
        weak_areas = []

        # Check response quality
        frustrated_ratio = self.analytics.query("""
            SELECT COUNT(*) FILTER (WHERE metric_name = 'sentiment_frustrated')
                   * 1.0 / NULLIF(COUNT(*), 0) as ratio
            FROM metrics
            WHERE ts > now() - INTERVAL 24 HOUR
            AND component = 'em'
        """)
        if frustrated_ratio > 0.2:
            weak_areas.append("High frustration rate (>20%) — review failed interactions")

        # Check recall quality
        miss_rate = self.analytics.query("""
            SELECT AVG(metric_value)
            FROM metrics
            WHERE metric_name = 'recall_miss_rate'
            AND ts > now() - INTERVAL 24 HOUR
        """)
        if miss_rate > 0.3:
            weak_areas.append("High recall miss rate (>30%) — knowledge gaps or index issues")

        # Check knowledge gaps
        new_gaps = self.analytics.query("""
            SELECT topic, ask_count
            FROM knowledge_gaps
            WHERE resolved = FALSE
            ORDER BY ask_count DESC
            LIMIT 5
        """)
        if new_gaps:
            weak_areas.append(f"Top unresolved knowledge gaps: {[g['topic'] for g in new_gaps]}")

        # Log reflection
        reflection = {
            "timestamp": now(),
            "performance_summary": report,
            "weak_areas": weak_areas,
            "total_interactions": report.get("total", 0),
        }
        self.analytics.log_metric("dmn", "self_reflection", 1, reflection)

        # If critical issues found, schedule remediation
        for area in weak_areas:
            await self._create_remediation_task(area)
```

---

#### 6.3.2 Spontaneous Planning (`control/dmn.py` → `_plan()`)

**What**: Anticipate future needs based on user patterns. Proactively prepare.

```python
    async def _plan(self):
        """Spontaneous planning: What should I prepare for?"""
        if not self.active: return

        # Pattern 1: User behavior prediction
        # "User_123 usually asks about churn on Mondays"
        user_patterns = self.analytics.query("""
            SELECT
                dimensions->>'user_id' as user_id,
                metric_name,
                EXTRACT(DOW FROM ts) as day_of_week,
                EXTRACT(HOUR FROM ts) as hour,
                COUNT(*) as frequency
            FROM metrics
            WHERE component = 'response'
            AND ts > now() - INTERVAL 30 DAY
            GROUP BY user_id, metric_name, day_of_week, hour
            HAVING COUNT(*) >= 3
            ORDER BY frequency DESC
        """)

        for pattern in user_patterns:
            # Pre-cache relevant data before the user typically asks
            await self._schedule_pre_cache(pattern)

        # Pattern 2: Data freshness
        # "Billing data refreshes monthly — schedule re-ingestion"
        stale_docs = self.analytics.query("""
            SELECT id, title, file_path, ingested_at
            FROM long_form_documents
            WHERE ingested_at < now() - INTERVAL 30 DAY
            AND source = 'admin_upload'
        """)
        if stale_docs:
            self.analytics.log_metric("dmn", "stale_docs_found", len(stale_docs))

        # Pattern 3: Performance optimization
        # "Collections queries are slow — optimize those indexes"
        slow_domains = self.analytics.query("""
            SELECT dimensions->>'domain' as domain,
                   AVG(metric_value) as avg_latency
            FROM metrics
            WHERE metric_name = 'recall_latency_ms'
            AND ts > now() - INTERVAL 7 DAY
            GROUP BY domain
            HAVING AVG(metric_value) > 100
        """)
        for domain in slow_domains:
            await self._schedule_index_optimization(domain["domain"])

    async def _schedule_pre_cache(self, pattern: dict):
        """Schedule a pre-cache task via PM for predicted user needs."""
        # Calculate next expected query time
        target_day = pattern["day_of_week"]
        target_hour = pattern["hour"]
        pre_cache_time = self._next_occurrence(target_day, target_hour - 1)  # 1 hour before

        await self.pm.create_schedule({
            "task_name": f"pre_cache_{pattern['user_id']}_{pattern['metric_name']}",
            "task_type": "pre_cache",
            "scheduled_at": pre_cache_time,
            "payload": {"user_id": pattern["user_id"], "query_type": pattern["metric_name"]},
            "cron_expr": None,  # one-shot, re-evaluated each DMN cycle
        })

    async def _schedule_index_optimization(self, domain: str):
        """Schedule FAISS index optimization for a slow domain."""
        await self.pm.create_schedule({
            "task_name": f"optimize_{domain}_indexes",
            "task_type": "maintenance",
            "scheduled_at": self._next_sleep_window(),
            "payload": {"domain": domain, "action": "reindex_faiss"},
        })

    async def _create_remediation_task(self, weak_area: str):
        """Create a PM task to address an identified weakness."""
        await self.pm.create_schedule({
            "task_name": f"remediate_{hash(weak_area) % 10000}",
            "task_type": "remediation",
            "scheduled_at": self._next_sleep_window(),
            "payload": {"issue": weak_area, "action": "investigate_and_fix"},
        })
```

---

#### 6.3.3 Maintenance (delegates to MML)

Phase 3 delegates to MML's existing sleep tasks (Section 4.2):
- Consolidation, forgetting, coherence scan, conflict resolution
- Index optimization, integrity checks, cross-memory linking
- Learning tasks (generalization, abstraction, etc.)

The key addition: DMN **prioritizes maintenance based on self-reflection results**. If reflection found "high recall miss rate" → prioritize index optimization over forgetting.

```python
    async def _prioritize_maintenance(self, weak_areas: list[str]) -> list[str]:
        """Reorder MML sleep tasks based on reflection findings."""
        priority_map = {
            "recall_miss": ["optimization", "consolidation", "linking"],
            "frustration": ["conflict", "coherence", "consolidation"],
            "knowledge_gaps": ["consolidation", "linking"],
            "slow_queries": ["optimization"],
        }
        boosted = []
        for area in weak_areas:
            for keyword, tasks in priority_map.items():
                if keyword in area.lower():
                    boosted.extend(tasks)
        return boosted  # MML runs these first
```

---

### 6.4 Control Systems — Interaction Flow

```
                    ┌────────────────────┐
                    │  SALIENCE NETWORK   │ ← events from all systems
                    │  (priority scoring) │
                    └────────┬───────────┘
                             ↓
                    ┌────────────────────┐
                    │  CENTRAL EXECUTIVE  │
                    │  (global event loop)│
                    └────────┬───────────┘
                             │
              ┌──────────────┼──────────────────┐
              ↓              ↓                   ↓
        ┌───────────┐  ┌──────────┐  ┌───────────────────┐
        │ Response   │  │ PM       │  │ Default Mode      │
        │ Layer      │  │ Scheduler│  │ Network           │
        │            │  │          │  │                   │
        │ (active    │  │ (tasks)  │  │ (idle mode)       │
        │  mode)     │  │          │  │ - Self-reflection │
        └─────┬─────┘  └────┬─────┘  │ - Planning        │
              │              │        │ - Maintenance     │
              ↓              ↓        └─────────┬─────────┘
        ┌──────────────────────────────────────┐│
        │         Memory Layer (MML)            ││
        │  (recall, learning, maintenance)     ←┘
        └──────────────────────────────────────┘
              ↓
        ┌──────────────────────────────────────┐
        │         Analytics (DuckDB)            │
        │  (all systems log metrics here)       │
        └──────────────────────────────────────┘
```

**Lifecycle**:

```
Agent starts
    ↓
CEN initializes all systems
    ↓
Salience Network starts listening for events
    ↓
Active mode: CEN routes user requests → Response Layer
    ↓
5 min idle → Salience detects → CEN switches to idle
    ↓
DMN activates: reflect → plan → maintain
    ↓
User message arrives → Salience interrupts → CEN switches to active
    ↓
DMN gracefully stops → Response Layer handles request
    ↓
... (cycle continues)
```

---

### 6.4 Governor (`control/governor.py`)

**Purpose**: The single enforcement point for ALL guardrails in the system. Every expensive action — LLM calls, sub-agent spawns, sandbox executions, data queries, memory writes — goes through the Governor before it happens. If the Governor says no, it doesn't happen.

**Key Properties**:
- **One place** — Every limit, cap, and circuit breaker defined in one class. One glance → entire guardrail surface.
- **Non-bypassable** — CEN routes through Governor. No module calls connectors directly.
- **Graceful degradation** — Denied actions get a reason and a suggested alternative (use cheaper model, queue for later, ask user).
- **Real-time tracking** — Counters, budgets, and circuit breaker states updated on every action.

**The Complete Guardrail Surface**:

```python
from dataclasses import dataclass, field
from collections import defaultdict
import time
import asyncio
from enum import Enum


class DenialReason(Enum):
    RATE_LIMITED = "rate_limited"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MAX_CONCURRENT = "max_concurrent"
    SESSION_LIMIT = "session_limit"
    CIRCUIT_OPEN = "circuit_open"
    REQUIRES_APPROVAL = "requires_approval"


@dataclass
class GuardrailVerdict:
    allowed: bool
    reason: DenialReason | None = None
    message: str = ""
    suggestion: str = ""      # "use cheaper model", "queue for later", etc.
    wait_seconds: float = 0   # how long until this action would be allowed


@dataclass
class GovernorLimits:
    """ALL guardrails in one place. Change these → change the system's behavior."""

    # ── LLM ──
    llm_calls_per_minute: int = 20
    llm_calls_per_hour: int = 200
    llm_calls_per_day: int = 2000
    llm_daily_budget_usd: float = 10.0
    llm_max_tokens_per_call: int = 4096
    llm_max_concurrent: int = 5

    # ── Sub-agents ──
    sub_agents_concurrent: int = 10
    sub_agents_per_session: int = 50
    sub_agents_per_hour: int = 30

    # ── Sandbox ──
    sandbox_runs_per_hour: int = 20
    sandbox_max_memory_mb: int = 512
    sandbox_max_concurrent: int = 3

    # ── Data Queries ──
    data_queries_per_hour: dict = field(default_factory=lambda: {
        "snowflake": 50,
        "databricks": 50,
        "azure_sql": 50,
    })
    data_max_rows_per_query: int = 10_000

    # ── Memory Writes ──
    memory_writes_per_hour: dict = field(default_factory=lambda: {
        "sfm": 100,
        "lfm": 20,
        "am": 50,
        "em": 50,
        "mm": 20,
        "wm": 500,
    })

    # ── Concurrent Requests ──
    max_concurrent_user_requests: int = 5

    # ── Circuit Breaker ──
    circuit_breaker_failure_threshold: int = 5     # failures in window → open circuit
    circuit_breaker_window_seconds: int = 60       # failure counting window
    circuit_breaker_cooldown_seconds: int = 300    # how long circuit stays open

    # ── Human-in-the-loop Gates ──
    actions_requiring_approval: frozenset = frozenset({
        "delete_memory",
        "modify_procedure",
        "execute_unknown_code",
        "write_to_external_db",
        "bulk_memory_operation",
    })
```

**Governor Implementation**:

```python
class Governor:
    def __init__(self, limits: GovernorLimits | None = None):
        self.limits = limits or GovernorLimits()

        # ── Rate tracking (sliding windows) ──
        self._call_timestamps: dict[str, list[float]] = defaultdict(list)

        # ── Concurrent tracking ──
        self._concurrent: dict[str, int] = defaultdict(int)

        # ── Budget tracking ──
        self._daily_spend_usd: float = 0.0
        self._daily_spend_reset: float = self._start_of_day()

        # ── Session tracking ──
        self._session_counts: dict[str, int] = defaultdict(int)

        # ── Circuit breakers ──
        self._circuit_failures: dict[str, list[float]] = defaultdict(list)  # connector → [fail timestamps]
        self._circuit_open_until: dict[str, float] = {}                     # connector → reopen time

        # ── Approval queue ──
        self._pending_approvals: dict[str, asyncio.Future] = {}

    # ──────────────────────────────────────────────────────
    # Main entry point — everything calls this
    # ──────────────────────────────────────────────────────

    async def authorize(self, action: str, **kwargs) -> GuardrailVerdict:
        """Single entry point. Every action goes through here."""
        match action:
            case "llm_call":
                return self._check_llm(kwargs.get("estimated_cost_usd", 0.0))
            case "spawn_sub_agent":
                return self._check_sub_agent()
            case "sandbox_execute":
                return self._check_sandbox()
            case "data_query":
                return self._check_data_query(kwargs.get("connector", "unknown"))
            case "memory_write":
                return self._check_memory_write(kwargs.get("memory_type", "unknown"))
            case _:
                if action in self.limits.actions_requiring_approval:
                    return self._check_approval(action)
                return GuardrailVerdict(allowed=True)

    # ──────────────────────────────────────────────────────
    # LLM guardrails
    # ──────────────────────────────────────────────────────

    def _check_llm(self, estimated_cost: float) -> GuardrailVerdict:
        now = time.time()
        self._reset_daily_budget_if_needed(now)

        # Budget check
        if self._daily_spend_usd + estimated_cost > self.limits.llm_daily_budget_usd:
            remaining = self.limits.llm_daily_budget_usd - self._daily_spend_usd
            return GuardrailVerdict(
                allowed=False, reason=DenialReason.BUDGET_EXHAUSTED,
                message=f"Daily LLM budget exhausted. Spent: ${self._daily_spend_usd:.2f} / ${self.limits.llm_daily_budget_usd:.2f}",
                suggestion="Switch to cheaper model or wait until budget resets",
            )

        # Concurrent check
        if self._concurrent["llm"] >= self.limits.llm_max_concurrent:
            return GuardrailVerdict(
                allowed=False, reason=DenialReason.MAX_CONCURRENT,
                message=f"Max concurrent LLM calls reached ({self.limits.llm_max_concurrent})",
                suggestion="Queue and retry when a slot opens",
            )

        # Rate limit checks (per minute, per hour, per day)
        ts = self._call_timestamps["llm"]
        self._prune_timestamps(ts, now)

        calls_last_minute = sum(1 for t in ts if now - t < 60)
        if calls_last_minute >= self.limits.llm_calls_per_minute:
            return GuardrailVerdict(
                allowed=False, reason=DenialReason.RATE_LIMITED,
                message=f"LLM rate limit: {calls_last_minute}/{self.limits.llm_calls_per_minute} per minute",
                suggestion="Wait a few seconds",
                wait_seconds=60 - (now - ts[-self.limits.llm_calls_per_minute]),
            )

        calls_last_hour = sum(1 for t in ts if now - t < 3600)
        if calls_last_hour >= self.limits.llm_calls_per_hour:
            return GuardrailVerdict(
                allowed=False, reason=DenialReason.RATE_LIMITED,
                message=f"LLM rate limit: {calls_last_hour}/{self.limits.llm_calls_per_hour} per hour",
                suggestion="Use cheaper model or wait",
            )

        return GuardrailVerdict(allowed=True)

    # ──────────────────────────────────────────────────────
    # Sub-agent guardrails
    # ──────────────────────────────────────────────────────

    def _check_sub_agent(self) -> GuardrailVerdict:
        now = time.time()

        # Concurrent
        if self._concurrent["sub_agent"] >= self.limits.sub_agents_concurrent:
            return GuardrailVerdict(
                allowed=False, reason=DenialReason.MAX_CONCURRENT,
                message=f"Max concurrent sub-agents ({self.limits.sub_agents_concurrent})",
                suggestion="Wait for running sub-agents to complete",
            )

        # Session total
        if self._session_counts["sub_agent"] >= self.limits.sub_agents_per_session:
            return GuardrailVerdict(
                allowed=False, reason=DenialReason.SESSION_LIMIT,
                message=f"Session sub-agent limit ({self.limits.sub_agents_per_session})",
                suggestion="Consolidate remaining work into fewer agents",
            )

        # Hourly rate
        ts = self._call_timestamps["sub_agent"]
        self._prune_timestamps(ts, now)
        calls_last_hour = sum(1 for t in ts if now - t < 3600)
        if calls_last_hour >= self.limits.sub_agents_per_hour:
            return GuardrailVerdict(
                allowed=False, reason=DenialReason.RATE_LIMITED,
                message=f"Sub-agent hourly limit ({self.limits.sub_agents_per_hour}/hr)",
                suggestion="Queue and retry later",
            )

        return GuardrailVerdict(allowed=True)

    # ──────────────────────────────────────────────────────
    # Sandbox guardrails
    # ──────────────────────────────────────────────────────

    def _check_sandbox(self) -> GuardrailVerdict:
        now = time.time()

        if self._concurrent["sandbox"] >= self.limits.sandbox_max_concurrent:
            return GuardrailVerdict(
                allowed=False, reason=DenialReason.MAX_CONCURRENT,
                message=f"Max concurrent sandbox runs ({self.limits.sandbox_max_concurrent})",
                suggestion="Wait for current execution to finish",
            )

        ts = self._call_timestamps["sandbox"]
        self._prune_timestamps(ts, now)
        runs_last_hour = sum(1 for t in ts if now - t < 3600)
        if runs_last_hour >= self.limits.sandbox_runs_per_hour:
            return GuardrailVerdict(
                allowed=False, reason=DenialReason.RATE_LIMITED,
                message=f"Sandbox hourly limit ({self.limits.sandbox_runs_per_hour}/hr)",
                suggestion="Reduce code generation frequency",
            )

        return GuardrailVerdict(allowed=True)

    # ──────────────────────────────────────────────────────
    # Data query guardrails
    # ──────────────────────────────────────────────────────

    def _check_data_query(self, connector: str) -> GuardrailVerdict:
        now = time.time()

        # Circuit breaker
        if connector in self._circuit_open_until:
            if now < self._circuit_open_until[connector]:
                remaining = self._circuit_open_until[connector] - now
                return GuardrailVerdict(
                    allowed=False, reason=DenialReason.CIRCUIT_OPEN,
                    message=f"Circuit breaker OPEN for {connector}. Reopens in {remaining:.0f}s",
                    suggestion="Use cached data or alternative connector",
                    wait_seconds=remaining,
                )
            else:
                del self._circuit_open_until[connector]  # circuit closed, retry

        # Rate limit per connector
        limit = self.limits.data_queries_per_hour.get(connector, 50)
        ts = self._call_timestamps[f"data_{connector}"]
        self._prune_timestamps(ts, now)
        queries_last_hour = sum(1 for t in ts if now - t < 3600)
        if queries_last_hour >= limit:
            return GuardrailVerdict(
                allowed=False, reason=DenialReason.RATE_LIMITED,
                message=f"{connector} query limit ({limit}/hr)",
                suggestion="Use cached results or reduce query frequency",
            )

        return GuardrailVerdict(allowed=True)

    # ──────────────────────────────────────────────────────
    # Memory write guardrails
    # ──────────────────────────────────────────────────────

    def _check_memory_write(self, memory_type: str) -> GuardrailVerdict:
        now = time.time()
        limit = self.limits.memory_writes_per_hour.get(memory_type, 50)
        ts = self._call_timestamps[f"mem_{memory_type}"]
        self._prune_timestamps(ts, now)
        writes_last_hour = sum(1 for t in ts if now - t < 3600)
        if writes_last_hour >= limit:
            return GuardrailVerdict(
                allowed=False, reason=DenialReason.RATE_LIMITED,
                message=f"{memory_type} write limit ({limit}/hr)",
                suggestion="Batch writes or reduce frequency",
            )
        return GuardrailVerdict(allowed=True)

    # ──────────────────────────────────────────────────────
    # Human-in-the-loop gate
    # ──────────────────────────────────────────────────────

    def _check_approval(self, action: str) -> GuardrailVerdict:
        return GuardrailVerdict(
            allowed=False, reason=DenialReason.REQUIRES_APPROVAL,
            message=f"Action '{action}' requires human approval",
            suggestion="Request approval from user before proceeding",
        )

    # ──────────────────────────────────────────────────────
    # Lifecycle hooks — called by CEN after action completes
    # ──────────────────────────────────────────────────────

    def record_action(self, action: str, **kwargs):
        """Called AFTER a successful action to update counters."""
        now = time.time()
        category = self._action_to_category(action)
        self._call_timestamps[category].append(now)
        self._session_counts[category] = self._session_counts.get(category, 0) + 1

        if action == "llm_call" and "cost_usd" in kwargs:
            self._daily_spend_usd += kwargs["cost_usd"]

    def acquire(self, resource: str):
        """Mark a concurrent resource as in-use."""
        self._concurrent[resource] += 1

    def release(self, resource: str):
        """Mark a concurrent resource as free."""
        self._concurrent[resource] = max(0, self._concurrent[resource] - 1)

    def record_failure(self, connector: str):
        """Record a connector failure for circuit breaker."""
        now = time.time()
        failures = self._circuit_failures[connector]
        failures.append(now)
        # Count failures in window
        recent = [t for t in failures if now - t < self.limits.circuit_breaker_window_seconds]
        self._circuit_failures[connector] = recent
        if len(recent) >= self.limits.circuit_breaker_failure_threshold:
            self._circuit_open_until[connector] = now + self.limits.circuit_breaker_cooldown_seconds

    # ──────────────────────────────────────────────────────
    # Status — one glance at everything
    # ──────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return the full guardrail status. For dashboards, logging, diagnostics."""
        now = time.time()
        return {
            "llm": {
                "daily_spend_usd": round(self._daily_spend_usd, 4),
                "daily_budget_usd": self.limits.llm_daily_budget_usd,
                "budget_remaining_usd": round(self.limits.llm_daily_budget_usd - self._daily_spend_usd, 4),
                "concurrent": self._concurrent["llm"],
                "max_concurrent": self.limits.llm_max_concurrent,
                "calls_last_hour": sum(1 for t in self._call_timestamps["llm"] if now - t < 3600),
            },
            "sub_agents": {
                "concurrent": self._concurrent["sub_agent"],
                "max_concurrent": self.limits.sub_agents_concurrent,
                "session_total": self._session_counts.get("sub_agent", 0),
                "session_limit": self.limits.sub_agents_per_session,
            },
            "sandbox": {
                "concurrent": self._concurrent["sandbox"],
                "max_concurrent": self.limits.sandbox_max_concurrent,
                "runs_last_hour": sum(1 for t in self._call_timestamps["sandbox"] if now - t < 3600),
            },
            "circuit_breakers": {
                conn: {"open": now < reopen, "reopens_in": max(0, reopen - now)}
                for conn, reopen in self._circuit_open_until.items()
            },
        }

    # ──────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────

    def _prune_timestamps(self, ts: list[float], now: float):
        """Remove timestamps older than 1 day."""
        cutoff = now - 86400
        while ts and ts[0] < cutoff:
            ts.pop(0)

    def _reset_daily_budget_if_needed(self, now: float):
        if now - self._daily_spend_reset > 86400:
            self._daily_spend_usd = 0.0
            self._daily_spend_reset = self._start_of_day()

    @staticmethod
    def _start_of_day() -> float:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        return datetime(now.year, now.month, now.day, tzinfo=timezone.utc).timestamp()

    @staticmethod
    def _action_to_category(action: str) -> str:
        return {"llm_call": "llm", "spawn_sub_agent": "sub_agent",
                "sandbox_execute": "sandbox"}.get(action, action)
```

**How CEN uses the Governor**:

```python
# In cen.py — BEFORE dispatching any work
class CentralExecutiveNetwork:
    def __init__(self, ..., governor: Governor):
        self.governor = governor

    async def _handle_user_message(self, event):
        # Every LLM call goes through governor
        verdict = await self.governor.authorize("llm_call", estimated_cost_usd=0.03)
        if not verdict.allowed:
            # Degrade gracefully
            if verdict.reason == DenialReason.BUDGET_EXHAUSTED:
                return self._budget_exhausted_response(verdict)
            elif verdict.reason == DenialReason.RATE_LIMITED:
                await asyncio.sleep(verdict.wait_seconds)
                return await self._handle_user_message(event)  # retry
            else:
                return self._denial_response(verdict)

        self.governor.acquire("llm")
        try:
            response = await self.response.handle(event.payload)
            self.governor.record_action("llm_call", cost_usd=response.cost)
        except Exception as e:
            self.governor.record_failure("llm")
            raise
        finally:
            self.governor.release("llm")

# In response/sub_agents.py — BEFORE spawning
async def spawn(self, task):
    verdict = await governor.authorize("spawn_sub_agent")
    if not verdict.allowed:
        return self._handle_denial(verdict)
    governor.acquire("sub_agent")
    try:
        result = await self._run_sub_agent(task)
        governor.record_action("spawn_sub_agent")
        return result
    finally:
        governor.release("sub_agent")
```

**Governor → Connector failure flow** (circuit breaker):

```
Skills Engine → governor.authorize("data_query", connector="snowflake") → allowed
    ↓
Snowflake query fails
    ↓
governor.record_failure("snowflake")   ← failure #1
    ... (4 more failures within 60s)
governor.record_failure("snowflake")   ← failure #5 → CIRCUIT OPENS
    ↓
Next query: governor.authorize("data_query", connector="snowflake")
    → DENIED: "Circuit breaker OPEN. Reopens in 300s"
    → suggestion: "Use cached data or alternative connector"
    ↓
After 5 min cooldown → circuit closes → queries allowed again
```

---

### 6.5 Design Decisions

| Decision | Rationale |
|---|---|
| Salience as rule-based scoring, not LLM | Must be instant (<1ms). LLM adds latency to every event. Simple math is sufficient for priority. |
| CEN as single event loop | One coordination point prevents race conditions. asyncio handles concurrency without threads. |
| DMN interruptible | User requests must never wait for background tasks. Instant switch from idle → active. |
| DMN self-reflection uses analytics | No LLM needed — pure SQL queries over DuckDB. Free, fast, data-driven. |
| DMN spontaneous planning creates PM tasks | Reuses existing scheduling infrastructure. No new scheduler needed. |
| Resource reservation (1 slot for users) | Scheduled tasks never starve user requests. Users always have priority. |
| Frustration → escalation | Sentiment signal from EM triggers resource reallocation. Immediate quality improvement. |
| Backpressure over rejection | Under load, queue and slow down rather than drop requests. |
| Governor as single enforcement point | All guardrails in one class. One place to audit limits. No scattered rate-limit logic across modules. |
| Circuit breaker pattern | Auto-disable failing connectors instead of hammering them. Self-healing after cooldown. |
| Human gates for destructive ops | Prevents AI from deleting data or modifying procedures without explicit approval. |
| `GovernorLimits` as a dataclass | All caps visible in one place. Change one number → change system behavior. No hunting through files. |
| Governor separate from Audit | Governor = enforcement (should this happen?). Audit = recording (what happened). Different concerns. |

---

### 6.6 Codebase Structure

```
control/                             ← Control Systems (separate from memory/ and response/)
├── __init__.py
├── salience.py                      ← Salience Network (priority scoring, mode detection)
├── cen.py                           ← Central Executive Network (global event loop, resource mgmt)
├── dmn.py                           ← Default Mode Network (self-reflection, planning, maintenance)
└── governor.py                      ← Governor (guardrails, rate limits, budgets, circuit breakers)
```

---

## 7. Connectors Layer

The integration layer — all external interfaces (data sources, AI models, code execution) behind a unified connector contract. Every other layer imports from here. **No other module reads `os.environ` or manages credentials directly.**

`config.py` is the single source of truth for all environment variables, credentials, paths, model configs, and security policy.

---

### 7.1 Central Configuration (`connectors/config.py`)

**Purpose**: One place for ALL configuration. Every module in the agent imports from `config.py` instead of reading env vars, hardcoding paths, or managing credentials.

**What it owns**:

| Category | Examples |
|---|---|
| Storage paths | `/datadrive/sqlite/`, `/datadrive/faiss/`, `/datadrive/duckdb/`, `/datadrive/kuzu/` |
| DB connections | Snowflake account/warehouse/db, Databricks host/token, Azure SQL connection string |
| AI model configs | LLM provider/model/API key, embedding model path, NLI model path |
| Redis | Host, port, password, DB index, max memory |
| Budget & limits | LLM daily budget, max concurrent requests, max sub-agents |
| Sandbox policy | Allowed imports, blocked modules, timeouts, max memory, working directory |
| Feature flags | Enable/disable specific connectors, toggle debug logging |

**Implementation**:

```python
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class StorageConfig:
    """All /datadrive/ paths — one place."""
    base: Path = Path(os.getenv("DATADRIVE_PATH", "/datadrive"))

    @property
    def sqlite_dir(self) -> Path: return self.base / "sqlite"
    @property
    def hot_db(self) -> Path: return self.sqlite_dir / "hot.db"
    @property
    def cold_db(self) -> Path: return self.sqlite_dir / "cold.db"
    @property
    def duckdb_path(self) -> Path: return self.base / "duckdb" / "analytics.duckdb"
    @property
    def redis_dir(self) -> Path: return self.base / "redis"
    @property
    def kuzu_dir(self) -> Path: return self.base / "kuzu" / "graph"
    @property
    def faiss_dir(self) -> Path: return self.base / "faiss"
    @property
    def jsonl_dir(self) -> Path: return self.base / "jsonl"
    @property
    def md_dir(self) -> Path: return self.base / "md"
    @property
    def yaml_dir(self) -> Path: return self.base / "yaml"
    @property
    def sandbox_tmp(self) -> Path: return self.base / "sandbox" / "tmp"
    @property
    def models_dir(self) -> Path: return self.base / "models"


@dataclass(frozen=True)
class RedisConfig:
    host: str = os.getenv("REDIS_HOST", "127.0.0.1")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    password: str | None = os.getenv("REDIS_PASSWORD")
    db: int = int(os.getenv("REDIS_DB", "0"))
    max_memory_mb: int = int(os.getenv("REDIS_MAX_MEMORY_MB", "512"))
    ttl_inactivity_hours: int = 24


@dataclass(frozen=True)
class LLMConfig:
    provider: str = os.getenv("LLM_PROVIDER", "openai")          # "openai" | "anthropic" | "local"
    model: str = os.getenv("LLM_MODEL", "gpt-4o")
    api_key: str | None = os.getenv("LLM_API_KEY")
    api_base: str | None = os.getenv("LLM_API_BASE")             # for local/custom endpoints
    cheap_model: str = os.getenv("LLM_CHEAP_MODEL", "gpt-4o-mini")  # fallback under budget pressure
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    daily_budget_usd: float = float(os.getenv("LLM_DAILY_BUDGET_USD", "10.0"))


@dataclass(frozen=True)
class EmbeddingConfig:
    """Pluggable embedding model configuration.
    
    Supported models (set via EMBEDDING_MODEL env var):
    - BAAI/bge-large-en-v1.5 (default, 1024 dims, best quality)
    - BAAI/bge-base-en-v1.5 (768 dims, faster)
    - nomic-embed-text-v1.5 (768 dims, good balance)
    - text-embedding-3-small (OpenAI API, 1536 dims)
    - text-embedding-3-large (OpenAI API, 3072 dims)
    
    Dimension is auto-detected from model at startup.
    """
    model_name: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    model_path: str | None = os.getenv("EMBEDDING_MODEL_PATH")   # local path override
    dimension: int | None = None  # auto-detected from model at connect time
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    use_api: bool = os.getenv("EMBEDDING_USE_API", "false").lower() == "true"  # True for OpenAI embeddings
    api_key: str | None = os.getenv("EMBEDDING_API_KEY")  # for API-based embeddings


@dataclass(frozen=True)
class NLIConfig:
    model_name: str = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
    model_path: str | None = os.getenv("NLI_MODEL_PATH")
    contradiction_threshold: float = 0.85
    entailment_threshold: float = 0.80


@dataclass(frozen=True)
class SandboxConfig:
    python_timeout_seconds: int = int(os.getenv("SANDBOX_PYTHON_TIMEOUT", "30"))
    shell_timeout_seconds: int = int(os.getenv("SANDBOX_SHELL_TIMEOUT", "15"))
    max_memory_mb: int = int(os.getenv("SANDBOX_MAX_MEMORY_MB", "512"))
    allowed_imports: frozenset[str] = frozenset({
        "pandas", "numpy", "json", "datetime", "math",
        "collections", "itertools", "functools", "re",
        "csv", "decimal", "statistics", "textwrap",
    })
    blocked_modules: frozenset[str] = frozenset({
        "os", "sys", "subprocess", "shutil", "socket",
        "http", "urllib", "importlib", "__builtins__",
        "ctypes", "multiprocessing", "signal", "threading",
    })
    shell_allowed_commands: frozenset[str] = frozenset({
        "cat", "head", "tail", "wc", "grep", "awk",
        "sort", "cut", "jq", "echo", "date", "tr",
    })


@dataclass(frozen=True)
class SnowflakeConfig:
    account: str | None = os.getenv("SNOWFLAKE_ACCOUNT")
    user: str | None = os.getenv("SNOWFLAKE_USER")
    password: str | None = os.getenv("SNOWFLAKE_PASSWORD")
    warehouse: str | None = os.getenv("SNOWFLAKE_WAREHOUSE")
    database: str | None = os.getenv("SNOWFLAKE_DATABASE")
    schema: str = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
    role: str | None = os.getenv("SNOWFLAKE_ROLE")


@dataclass(frozen=True)
class DatabricksConfig:
    host: str | None = os.getenv("DATABRICKS_HOST")
    token: str | None = os.getenv("DATABRICKS_TOKEN")
    http_path: str | None = os.getenv("DATABRICKS_HTTP_PATH")
    catalog: str | None = os.getenv("DATABRICKS_CATALOG")
    schema: str = os.getenv("DATABRICKS_SCHEMA", "default")


@dataclass(frozen=True)
class AzureSQLConfig:
    connection_string: str | None = os.getenv("AZURE_SQL_CONNECTION_STRING")
    server: str | None = os.getenv("AZURE_SQL_SERVER")
    database: str | None = os.getenv("AZURE_SQL_DATABASE")
    driver: str = os.getenv("AZURE_SQL_DRIVER", "ODBC Driver 18 for SQL Server")


@dataclass(frozen=True)
class AgentConfig:
    """Top-level config — the single import every module uses."""
    max_concurrent_requests: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
    max_sub_agents: int = int(os.getenv("MAX_SUB_AGENTS", "10"))
    idle_timeout_seconds: int = int(os.getenv("IDLE_TIMEOUT_SECONDS", "300"))
    debug: bool = os.getenv("AGENT_DEBUG", "false").lower() == "true"

    storage: StorageConfig = field(default_factory=StorageConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    nli: NLIConfig = field(default_factory=NLIConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    snowflake: SnowflakeConfig = field(default_factory=SnowflakeConfig)
    databricks: DatabricksConfig = field(default_factory=DatabricksConfig)
    azure_sql: AzureSQLConfig = field(default_factory=AzureSQLConfig)


# ── Singleton ──────────────────────────────────────────────
# Every module does: from connectors.config import config
config = AgentConfig()
```

**Usage everywhere**:

```python
# In memory/types/wm.py
from connectors.config import config
redis_client = Redis(host=config.redis.host, port=config.redis.port)

# In memory/management/recall.py
from connectors.config import config
embedder = SentenceTransformer(config.embedding.model_name)
index_dir = config.storage.faiss_dir

# In response/thinking.py
from connectors.config import config
llm = LLMConnector(provider=config.llm.provider, model=config.llm.model)

# In response/skills/code_gen.py
from connectors.config import config
sandbox = PythonSandbox(
    timeout=config.sandbox.python_timeout_seconds,
    allowed_imports=config.sandbox.allowed_imports,
)

# In analytics/analytics.py
from connectors.config import config
db = duckdb.connect(str(config.storage.duckdb_path))
```

**Design Decisions**:

| Decision | Rationale |
|---|---|
| `frozen=True` dataclasses | Config is read-once-at-startup, immutable at runtime. Prevents accidental mutation. |
| `os.getenv` with defaults | Works with `.env` files, Docker env, systemd, or plain shell exports. No framework dependency. |
| Singleton `config` at module level | One import, one object. No dependency injection boilerplate. |
| Nested dataclasses | Organized by concern (`config.llm.model` not `config.llm_model`). Clean namespacing. |
| Credentials via env vars | Never in code, never in config files committed to git. Standard 12-factor approach. |
| `frozenset` for sandbox policy | Immutable sets — can't accidentally add `os` to allowed imports at runtime. |

---

### 7.2 Connector Base Interface (`connectors/base.py`)

**Purpose**: Common contract for all connectors. Every data connector, AI connector, and sandbox implements this interface.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class ConnectorType(Enum):
    DATA = "data"
    AI = "ai"
    SANDBOX = "sandbox"


class ConnectorStatus(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    NOT_CONFIGURED = "not_configured"


@dataclass
class ConnectorInfo:
    name: str                    # "snowflake", "llm", "python_sandbox"
    connector_type: ConnectorType
    status: ConnectorStatus
    metadata: dict               # connector-specific info (version, region, etc.)


class BaseConnector(ABC):
    """Every connector implements this contract."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection. Called once at startup or on first use."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Clean shutdown. Release pools, close connections."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Is this connector alive and responsive?"""
        ...

    @abstractmethod
    def info(self) -> ConnectorInfo:
        """Return connector metadata for registry and diagnostics."""
        ...

    def is_configured(self) -> bool:
        """Check if required env vars are set. Override per connector."""
        return True
```

---

### 7.3 Data Connectors (`connectors/data/`)

**Purpose**: Interface to external data sources. The Skills Engine calls these to fetch or write domain-specific data. All implement `BaseConnector` + a query interface.

#### 7.3.1 Data Connector Interface

```python
class BaseDataConnector(BaseConnector):
    """Extended interface for data source connectors."""
    connector_type = ConnectorType.DATA

    @abstractmethod
    async def execute_query(self, query: str, params: dict | None = None) -> list[dict]:
        """Execute a parameterized query. Returns list of row dicts."""
        ...

    @abstractmethod
    async def execute_query_df(self, query: str, params: dict | None = None):
        """Execute query and return a pandas DataFrame."""
        ...

    @abstractmethod
    def get_schema(self, table: str | None = None) -> dict:
        """Return schema info for introspection (tables, columns, types)."""
        ...
```

#### 7.3.2 Snowflake Connector (`connectors/data/snowflake.py`)

```python
import asyncio
import snowflake.connector
from config import config
from connectors.base import BaseDataConnector, ConnectorInfo, ConnectorStatus


class SnowflakeConnector(BaseDataConnector):
    """Async Snowflake connector. Wraps sync driver with asyncio.to_thread()."""

    def __init__(self):
        self._conn = None
        self._pool = None

    def is_configured(self) -> bool:
        return all([config.snowflake.account, config.snowflake.user, config.snowflake.password])

    async def connect(self):
        if not self.is_configured():
            return
        self._conn = await asyncio.to_thread(
            snowflake.connector.connect,
            account=config.snowflake.account,
            user=config.snowflake.user,
            password=config.snowflake.password,
            warehouse=config.snowflake.warehouse,
            database=config.snowflake.database,
            schema=config.snowflake.schema,
            role=config.snowflake.role,
        )

    async def disconnect(self):
        if self._conn:
            await asyncio.to_thread(self._conn.close)

    async def health_check(self) -> bool:
        try:
            def _check():
                cur = self._conn.cursor()
                cur.execute("SELECT 1")
                return True
            return await asyncio.to_thread(_check)
        except Exception:
            return False

    async def execute_query(self, query: str, params: dict | None = None) -> list[dict]:
        def _execute():
            cur = self._conn.cursor(snowflake.connector.DictCursor)
            cur.execute(query, params or {})
            return cur.fetchall()
        return await asyncio.to_thread(_execute)

    async def execute_query_df(self, query: str, params: dict | None = None):
        def _execute():
            cur = self._conn.cursor()
            cur.execute(query, params or {})
            return cur.fetch_pandas_all()
        return await asyncio.to_thread(_execute)

    async def get_schema(self, table: str | None = None) -> dict:
        def _get():
            if table:
                rows = self._conn.cursor().execute(f"DESCRIBE TABLE {table}").fetchall()
                return {"table": table, "columns": [{"name": r[0], "type": r[1]} for r in rows]}
            rows = self._conn.cursor().execute("SHOW TABLES").fetchall()
            return {"tables": [r[1] for r in rows]}
        return await asyncio.to_thread(_get)

    def info(self) -> ConnectorInfo:
        return ConnectorInfo(
            name="snowflake",
            connector_type=self.connector_type,
            status=ConnectorStatus.CONNECTED if self._conn else ConnectorStatus.DISCONNECTED,
            metadata={"account": config.snowflake.account, "warehouse": config.snowflake.warehouse},
        )
```

#### 7.3.3 Databricks Connector (`connectors/data/databricks.py`)

```python
import asyncio
from databricks import sql as databricks_sql
from config import config
from connectors.base import BaseDataConnector, ConnectorInfo, ConnectorStatus


class DatabricksConnector(BaseDataConnector):
    """Async Databricks connector. Wraps sync driver with asyncio.to_thread()."""

    def __init__(self):
        self._conn = None

    def is_configured(self) -> bool:
        return all([config.databricks.host, config.databricks.token, config.databricks.http_path])

    async def connect(self):
        if not self.is_configured():
            return
        self._conn = await asyncio.to_thread(
            databricks_sql.connect,
            server_hostname=config.databricks.host,
            http_path=config.databricks.http_path,
            access_token=config.databricks.token,
            catalog=config.databricks.catalog,
            schema=config.databricks.schema,
        )

    async def disconnect(self):
        if self._conn:
            await asyncio.to_thread(self._conn.close)

    async def health_check(self) -> bool:
        try:
            def _check():
                cur = self._conn.cursor()
                cur.execute("SELECT 1")
                return True
            return await asyncio.to_thread(_check)
        except Exception:
            return False

    async def execute_query(self, query: str, params: dict | None = None) -> list[dict]:
        def _execute():
            cur = self._conn.cursor()
            cur.execute(query, params)
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]
        return await asyncio.to_thread(_execute)

    async def execute_query_df(self, query: str, params: dict | None = None):
        import pandas as pd
        rows = await self.execute_query(query, params)
        return pd.DataFrame(rows)

    async def get_schema(self, table: str | None = None) -> dict:
        def _get():
            cur = self._conn.cursor()
            if table:
                cur.execute(f"DESCRIBE TABLE {table}")
                return {"table": table, "columns": [{"name": r[0], "type": r[1]} for r in cur.fetchall()]}
            cur.execute("SHOW TABLES")
            return {"tables": [r[1] for r in cur.fetchall()]}
        return await asyncio.to_thread(_get)

    def info(self) -> ConnectorInfo:
        return ConnectorInfo(
            name="databricks",
            connector_type=self.connector_type,
            status=ConnectorStatus.CONNECTED if self._conn else ConnectorStatus.DISCONNECTED,
            metadata={"host": config.databricks.host, "catalog": config.databricks.catalog},
        )
```

#### 7.3.4 Azure SQL Connector (`connectors/data/azure_sql.py`)

```python
import asyncio
import pyodbc
from config import config
from connectors.base import BaseDataConnector, ConnectorInfo, ConnectorStatus


class AzureSQLConnector(BaseDataConnector):
    """Async Azure SQL connector. Wraps sync pyodbc driver with asyncio.to_thread()."""

    def __init__(self):
        self._conn = None

    def is_configured(self) -> bool:
        return config.azure_sql.connection_string is not None or (
            config.azure_sql.server is not None and config.azure_sql.database is not None
        )

    async def connect(self):
        if not self.is_configured():
            return
        def _connect():
            if config.azure_sql.connection_string:
                return pyodbc.connect(config.azure_sql.connection_string)
            else:
                conn_str = (
                    f"DRIVER={{{config.azure_sql.driver}}};"
                    f"SERVER={config.azure_sql.server};"
                    f"DATABASE={config.azure_sql.database};"
                    "Authentication=ActiveDirectoryInteractive;"
                )
                return pyodbc.connect(conn_str)
        self._conn = await asyncio.to_thread(_connect)

    async def disconnect(self):
        if self._conn:
            await asyncio.to_thread(self._conn.close)

    async def health_check(self) -> bool:
        try:
            def _check():
                self._conn.cursor().execute("SELECT 1").fetchone()
                return True
            return await asyncio.to_thread(_check)
        except Exception:
            return False

    async def execute_query(self, query: str, params: dict | None = None) -> list[dict]:
        def _execute():
            cur = self._conn.cursor()
            if params:
                cur.execute(query, list(params.values()))
            else:
                cur.execute(query)
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]
        return await asyncio.to_thread(_execute)

    async def execute_query_df(self, query: str, params: dict | None = None):
        import pandas as pd
        rows = await self.execute_query(query, params)
        return pd.DataFrame(rows)

    async def get_schema(self, table: str | None = None) -> dict:
        def _get():
            cur = self._conn.cursor()
            if table:
                cur.execute(
                    "SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = ?",
                    [table]
                )
                return {"table": table, "columns": [{"name": r[0], "type": r[1]} for r in cur.fetchall()]}
            cur.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'")
            return {"tables": [r[0] for r in cur.fetchall()]}
        return await asyncio.to_thread(_get)

    def info(self) -> ConnectorInfo:
        return ConnectorInfo(
            name="azure_sql",
            connector_type=self.connector_type,
            status=ConnectorStatus.CONNECTED if self._conn else ConnectorStatus.DISCONNECTED,
            metadata={"server": config.azure_sql.server, "database": config.azure_sql.database},
        )
```

#### 7.3.5 REST API Connector (`connectors/data/rest_api.py`)

```python
import aiohttp
from config import config
from connectors.base import BaseConnector, ConnectorInfo, ConnectorType, ConnectorStatus


class RESTAPIConnector(BaseConnector):
    """Generic REST API client with retry, auth, and timeout."""

    def __init__(self, name: str, base_url: str, auth_header: dict | None = None):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.auth_header = auth_header or {}
        self._session: aiohttp.ClientSession | None = None

    async def connect(self):
        self._session = aiohttp.ClientSession(
            base_url=self.base_url,
            headers=self.auth_header,
            timeout=aiohttp.ClientTimeout(total=30),
        )

    async def disconnect(self):
        if self._session:
            await self._session.close()

    async def health_check(self) -> bool:
        try:
            async with self._session.get("/health") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def get(self, path: str, params: dict | None = None) -> dict:
        async with self._session.get(path, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def post(self, path: str, payload: dict) -> dict:
        async with self._session.post(path, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()

    def info(self) -> ConnectorInfo:
        return ConnectorInfo(
            name=self.name,
            connector_type=ConnectorType.DATA,
            status=ConnectorStatus.CONNECTED if self._session else ConnectorStatus.DISCONNECTED,
            metadata={"base_url": self.base_url},
        )
```

#### 7.3.6 File Connector (`connectors/data/file_connector.py`)

```python
from pathlib import Path
import csv, json
from connectors.base import BaseConnector, ConnectorInfo, ConnectorType, ConnectorStatus


class FileConnector(BaseConnector):
    """Read local files: CSV, JSON, JSONL, Markdown, plain text."""

    def __init__(self, base_dir: str | Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else None

    async def connect(self): pass    # no connection needed
    async def disconnect(self): pass

    async def health_check(self) -> bool:
        return self.base_dir is None or self.base_dir.exists()

    def read_csv(self, path: str | Path) -> list[dict]:
        with open(self._resolve(path), newline="") as f:
            return list(csv.DictReader(f))

    def read_json(self, path: str | Path) -> dict | list:
        with open(self._resolve(path)) as f:
            return json.load(f)

    def read_jsonl(self, path: str | Path) -> list[dict]:
        with open(self._resolve(path)) as f:
            return [json.loads(line) for line in f if line.strip()]

    def read_text(self, path: str | Path) -> str:
        with open(self._resolve(path)) as f:
            return f.read()

    def write_jsonl(self, path: str | Path, records: list[dict]) -> None:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    def _resolve(self, path: str | Path) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        if self.base_dir:
            return self.base_dir / p
        return p

    def info(self) -> ConnectorInfo:
        return ConnectorInfo(
            name="file",
            connector_type=ConnectorType.DATA,
            status=ConnectorStatus.CONNECTED,
            metadata={"base_dir": str(self.base_dir)},
        )
```

---

### 7.4 AI Connectors (`connectors/ai/`)

**Purpose**: Interface to AI models. The Response Layer, Memory Layer, and MML all call these. Abstracts away provider differences so swapping OpenAI → Anthropic → local is a config change, not a code change.

#### 7.4.1 LLM Connector (`connectors/ai/llm.py`)

```python
from connectors.config import config
from connectors.base import BaseConnector, ConnectorInfo, ConnectorType, ConnectorStatus


class LLMConnector(BaseConnector):
    """Unified LLM interface. Supports OpenAI, Anthropic, and local endpoints."""

    def __init__(self):
        self._client = None
        self._cheap_client = None

    async def connect(self):
        match config.llm.provider:
            case "openai":
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=config.llm.api_key, base_url=config.llm.api_base)
            case "anthropic":
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(api_key=config.llm.api_key)
            case "local":
                from openai import AsyncOpenAI  # local endpoints use OpenAI-compatible API
                self._client = AsyncOpenAI(api_key="not-needed", base_url=config.llm.api_base)

    async def disconnect(self):
        self._client = None

    async def health_check(self) -> bool:
        try:
            await self.generate("ping", max_tokens=5)
            return True
        except Exception:
            return False

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> str:
        """Generate text from a prompt. Provider-agnostic."""
        model = model or config.llm.model
        max_tokens = max_tokens or config.llm.max_tokens
        temperature = temperature if temperature is not None else config.llm.temperature

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if config.llm.provider == "anthropic":
            resp = await self._client.messages.create(
                model=model, max_tokens=max_tokens, temperature=temperature,
                system=system or "", messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        else:
            # OpenAI / local (OpenAI-compatible)
            resp = await self._client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
            )
            return resp.choices[0].message.content

    async def generate_cheap(self, prompt: str, system: str | None = None) -> str:
        """Use the cheaper model (budget pressure, gate, simple tasks)."""
        return await self.generate(prompt, system=system, model=config.llm.cheap_model)

    def info(self) -> ConnectorInfo:
        return ConnectorInfo(
            name="llm",
            connector_type=ConnectorType.AI,
            status=ConnectorStatus.CONNECTED if self._client else ConnectorStatus.DISCONNECTED,
            metadata={"provider": config.llm.provider, "model": config.llm.model},
        )
```

#### 7.4.2 Embedding Connector (`connectors/ai/embedder.py`)

```python
import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer
from config import config
from connectors.base import BaseConnector, ConnectorInfo, ConnectorType, ConnectorStatus


class EmbeddingConnector(BaseConnector):
    """Pluggable embedding model connector. Supports local models and API-based embeddings.
    
    Local models: sentence-transformers compatible (BAAI/bge-*, nomic-*, etc.)
    API models: OpenAI text-embedding-3-* via API
    """

    def __init__(self):
        self._model: SentenceTransformer | None = None
        self._api_client = None
        self._dimension: int | None = None

    async def connect(self):
        if config.embedding.use_api:
            # API-based embeddings (OpenAI)
            from openai import AsyncOpenAI
            self._api_client = AsyncOpenAI(api_key=config.embedding.api_key)
            # Dimension lookup for known models
            dim_map = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
            self._dimension = dim_map.get(config.embedding.model_name, 1536)
        else:
            # Local sentence-transformers model
            path = config.embedding.model_path or config.embedding.model_name
            self._model = await asyncio.to_thread(SentenceTransformer, path)
            self._dimension = self._model.get_sentence_embedding_dimension()

    async def disconnect(self):
        self._model = None
        self._api_client = None

    async def health_check(self) -> bool:
        try:
            await self.embed_async("test")
            return True
        except Exception:
            return False

    @property
    def dimension(self) -> int:
        """Return embedding dimension. Auto-detected from model."""
        return self._dimension

    async def embed_async(self, text: str) -> np.ndarray:
        """Embed a single text asynchronously."""
        if config.embedding.use_api:
            resp = await self._api_client.embeddings.create(
                model=config.embedding.model_name, input=text
            )
            return np.array(resp.data[0].embedding, dtype=np.float32)
        else:
            return await asyncio.to_thread(
                self._model.encode, text, normalize_embeddings=True
            )

    async def embed_batch_async(self, texts: list[str]) -> np.ndarray:
        """Embed a batch asynchronously. Returns (N, dim) float32 array."""
        if config.embedding.use_api:
            resp = await self._api_client.embeddings.create(
                model=config.embedding.model_name, input=texts
            )
            return np.array([d.embedding for d in resp.data], dtype=np.float32)
        else:
            return await asyncio.to_thread(
                self._model.encode,
                texts,
                batch_size=config.embedding.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

    # Sync wrappers for backward compatibility (use async versions in new code)
    def embed(self, text: str) -> np.ndarray:
        """Sync embed. Prefer embed_async in async contexts."""
        return asyncio.get_event_loop().run_until_complete(self.embed_async(text))

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Sync batch embed. Prefer embed_batch_async in async contexts."""
        return asyncio.get_event_loop().run_until_complete(self.embed_batch_async(texts))

    def info(self) -> ConnectorInfo:
        return ConnectorInfo(
            name="embedder",
            connector_type=ConnectorType.AI,
            status=ConnectorStatus.CONNECTED if self._model else ConnectorStatus.DISCONNECTED,
            metadata={"model": config.embedding.model_name, "dimension": config.embedding.dimension},
        )
```

#### 7.4.3 NLI Connector (`connectors/ai/nli.py`)

```python
from sentence_transformers import CrossEncoder
from connectors.config import config
from connectors.base import BaseConnector, ConnectorInfo, ConnectorType, ConnectorStatus


class NLIConnector(BaseConnector):
    """Natural Language Inference — contradiction & entailment detection."""

    LABELS = ["contradiction", "entailment", "neutral"]

    def __init__(self):
        self._model: CrossEncoder | None = None

    async def connect(self):
        path = config.nli.model_path or config.nli.model_name
        self._model = CrossEncoder(path)

    async def disconnect(self):
        self._model = None

    async def health_check(self) -> bool:
        try:
            self.predict("The sky is blue.", "The sky is red.")
            return True
        except Exception:
            return False

    def predict(self, premise: str, hypothesis: str) -> dict:
        """Return label scores: {contradiction: 0.95, entailment: 0.02, neutral: 0.03}."""
        scores = self._model.predict([(premise, hypothesis)])[0]
        return dict(zip(self.LABELS, scores.tolist()))

    def is_contradicting(self, a: str, b: str) -> bool:
        """Quick check: do these two statements contradict?"""
        result = self.predict(a, b)
        return result["contradiction"] >= config.nli.contradiction_threshold

    def is_entailing(self, a: str, b: str) -> bool:
        """Quick check: does a entail b?"""
        result = self.predict(a, b)
        return result["entailment"] >= config.nli.entailment_threshold

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[dict]:
        """Batch NLI prediction."""
        scores = self._model.predict(pairs)
        return [dict(zip(self.LABELS, s.tolist())) for s in scores]

    def info(self) -> ConnectorInfo:
        return ConnectorInfo(
            name="nli",
            connector_type=ConnectorType.AI,
            status=ConnectorStatus.CONNECTED if self._model else ConnectorStatus.DISCONNECTED,
            metadata={"model": config.nli.model_name},
        )
```

#### 7.4.4 Reranker Connector (`connectors/ai/reranker.py`) — Future

```python
from sentence_transformers import CrossEncoder
from connectors.base import BaseConnector, ConnectorInfo, ConnectorType, ConnectorStatus


class RerankerConnector(BaseConnector):
    """Cross-encoder reranker for hybrid search results. Future enhancement."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    async def connect(self):
        self._model = CrossEncoder(self.model_name)

    async def disconnect(self):
        self._model = None

    async def health_check(self) -> bool:
        return self._model is not None

    def rerank(self, query: str, documents: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        """Rerank documents by relevance to query. Returns [(original_index, score), ...]."""
        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def info(self) -> ConnectorInfo:
        return ConnectorInfo(
            name="reranker",
            connector_type=ConnectorType.AI,
            status=ConnectorStatus.CONNECTED if self._model else ConnectorStatus.DISCONNECTED,
            metadata={"model": self.model_name},
        )
```

---

### 7.5 Sandbox Connectors (`connectors/sandbox/`)

**Purpose**: Safe execution environments for AI-generated code. The Skills Engine sends code here; sandboxes validate, isolate, execute, and return results. Security policy comes from `config.py`.

#### 7.5.1 Sandbox Base Interface

```python
from abc import abstractmethod
from dataclasses import dataclass
from connectors.base import BaseConnector, ConnectorType


@dataclass
class SandboxResult:
    success: bool
    output: str              # stdout / return value
    error: str | None        # stderr / exception message
    execution_time_ms: float
    truncated: bool          # True if output was cut to size limit


class BaseSandbox(BaseConnector):
    connector_type = ConnectorType.SANDBOX

    @abstractmethod
    async def execute(self, code: str, context: dict | None = None) -> SandboxResult:
        """Execute code in a sandboxed environment."""
        ...
```

#### 7.5.2 Python Sandbox (`connectors/sandbox/python_sandbox.py`)

```python
import ast
import subprocess
import tempfile
import time
from pathlib import Path

from connectors.config import config
from connectors.sandbox.base import BaseSandbox, SandboxResult
from connectors.base import ConnectorInfo, ConnectorStatus


class PythonSandbox(BaseSandbox):
    """Execute Python code in an isolated subprocess with restricted imports."""

    async def connect(self): pass
    async def disconnect(self): pass

    async def health_check(self) -> bool:
        result = await self.execute("print('ok')")
        return result.success and result.output.strip() == "ok"

    async def execute(self, code: str, context: dict | None = None) -> SandboxResult:
        # 1. Static validation
        violation = self._validate(code)
        if violation:
            return SandboxResult(
                success=False, output="", error=f"Security violation: {violation}",
                execution_time_ms=0, truncated=False,
            )

        # 2. Write to temp file
        work_dir = config.storage.sandbox_tmp
        work_dir.mkdir(parents=True, exist_ok=True)
        script_path = work_dir / f"run_{int(time.time() * 1000)}.py"

        # 3. Inject context as variables
        preamble = ""
        if context:
            for key, value in context.items():
                preamble += f"{key} = {repr(value)}\n"

        script_path.write_text(preamble + code)

        # 4. Execute in subprocess with resource limits
        start = time.monotonic()
        try:
            result = subprocess.run(
                ["python", str(script_path)],
                capture_output=True,
                text=True,
                timeout=config.sandbox.python_timeout_seconds,
                cwd=str(work_dir),
                env={
                    "PATH": "/usr/bin:/usr/local/bin",
                    "HOME": str(work_dir),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
            elapsed = (time.monotonic() - start) * 1000

            output = result.stdout[:10_000]  # cap output at 10KB
            return SandboxResult(
                success=result.returncode == 0,
                output=output,
                error=result.stderr[:5_000] if result.stderr else None,
                execution_time_ms=elapsed,
                truncated=len(result.stdout) > 10_000,
            )
        except subprocess.TimeoutExpired:
            elapsed = (time.monotonic() - start) * 1000
            return SandboxResult(
                success=False, output="",
                error=f"Timeout after {config.sandbox.python_timeout_seconds}s",
                execution_time_ms=elapsed, truncated=False,
            )
        finally:
            script_path.unlink(missing_ok=True)

    def _validate(self, code: str) -> str | None:
        """Static analysis: check imports and dangerous calls."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"Syntax error: {e}"

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module in config.sandbox.blocked_modules:
                        return f"Blocked import: {module}"
                    if module not in config.sandbox.allowed_imports:
                        return f"Unapproved import: {module}"

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    if module in config.sandbox.blocked_modules:
                        return f"Blocked import: {module}"
                    if module not in config.sandbox.allowed_imports:
                        return f"Unapproved import: {module}"

            # Block eval/exec/compile
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec", "compile", "__import__"):
                    return f"Blocked builtin: {node.func.id}"

        return None  # all clean

    def info(self) -> ConnectorInfo:
        return ConnectorInfo(
            name="python_sandbox",
            connector_type=self.connector_type,
            status=ConnectorStatus.CONNECTED,
            metadata={"timeout": config.sandbox.python_timeout_seconds, "max_memory_mb": config.sandbox.max_memory_mb},
        )
```

#### 7.5.3 Shell Sandbox (`connectors/sandbox/shell_sandbox.py`)

```python
import subprocess
import shlex
import time

from connectors.config import config
from connectors.sandbox.base import BaseSandbox, SandboxResult
from connectors.base import ConnectorInfo, ConnectorStatus


class ShellSandbox(BaseSandbox):
    """Execute shell commands with restricted command set."""

    async def connect(self): pass
    async def disconnect(self): pass

    async def health_check(self) -> bool:
        result = await self.execute("echo ok")
        return result.success and result.output.strip() == "ok"

    async def execute(self, code: str, context: dict | None = None) -> SandboxResult:
        # 1. Validate commands
        violation = self._validate(code)
        if violation:
            return SandboxResult(
                success=False, output="", error=f"Security violation: {violation}",
                execution_time_ms=0, truncated=False,
            )

        # 2. Execute in subprocess
        work_dir = config.storage.sandbox_tmp
        work_dir.mkdir(parents=True, exist_ok=True)

        # Inject context as env vars
        env = {"PATH": "/usr/bin:/usr/local/bin", "HOME": str(work_dir)}
        if context:
            for key, value in context.items():
                env[key] = str(value)

        start = time.monotonic()
        try:
            result = subprocess.run(
                ["sh", "-c", code],
                capture_output=True, text=True,
                timeout=config.sandbox.shell_timeout_seconds,
                cwd=str(work_dir),
                env=env,
            )
            elapsed = (time.monotonic() - start) * 1000

            output = result.stdout[:10_000]
            return SandboxResult(
                success=result.returncode == 0,
                output=output,
                error=result.stderr[:5_000] if result.stderr else None,
                execution_time_ms=elapsed,
                truncated=len(result.stdout) > 10_000,
            )
        except subprocess.TimeoutExpired:
            elapsed = (time.monotonic() - start) * 1000
            return SandboxResult(
                success=False, output="",
                error=f"Timeout after {config.sandbox.shell_timeout_seconds}s",
                execution_time_ms=elapsed, truncated=False,
            )

    def _validate(self, code: str) -> str | None:
        """Check that only allowed commands are used."""
        try:
            tokens = shlex.split(code)
        except ValueError:
            return "Could not parse command"

        # Check each piped command
        commands = code.split("|")
        for cmd in commands:
            cmd = cmd.strip()
            if not cmd:
                continue
            first_token = shlex.split(cmd)[0]
            if first_token not in config.sandbox.shell_allowed_commands:
                return f"Blocked command: {first_token}. Allowed: {sorted(config.sandbox.shell_allowed_commands)}"

        # Block dangerous patterns
        dangerous = ["sudo", "rm ", "rm\t", "chmod", "chown", "mkfs", "dd ", ">/dev/", "wget", "curl"]
        for pattern in dangerous:
            if pattern in code:
                return f"Blocked dangerous pattern: {pattern.strip()}"

        return None

    def info(self) -> ConnectorInfo:
        return ConnectorInfo(
            name="shell_sandbox",
            connector_type=self.connector_type,
            status=ConnectorStatus.CONNECTED,
            metadata={"timeout": config.sandbox.shell_timeout_seconds},
        )
```

---

### 7.6 Connector Registry (`connectors/registry.py`)

**Purpose**: Central registry of all available connectors. CEN uses this at startup to initialize configured connectors. Diagnostics use it for health checks.

```python
from connectors.config import config
from connectors.base import BaseConnector, ConnectorInfo
from connectors.data.snowflake import SnowflakeConnector
from connectors.data.databricks import DatabricksConnector
from connectors.data.azure_sql import AzureSQLConnector
from connectors.data.rest_api import RESTAPIConnector
from connectors.data.file_connector import FileConnector
from connectors.ai.llm import LLMConnector
from connectors.ai.embedder import EmbeddingConnector
from connectors.ai.nli import NLIConnector
from connectors.ai.reranker import RerankerConnector
from connectors.sandbox.python_sandbox import PythonSandbox
from connectors.sandbox.shell_sandbox import ShellSandbox


class ConnectorRegistry:
    """Discover, initialize, and manage all connectors."""

    def __init__(self):
        self._connectors: dict[str, BaseConnector] = {}

    def register(self, name: str, connector: BaseConnector):
        self._connectors[name] = connector

    def get(self, name: str) -> BaseConnector:
        if name not in self._connectors:
            raise KeyError(f"Connector '{name}' not registered")
        return self._connectors[name]

    def list_all(self) -> list[ConnectorInfo]:
        return [c.info() for c in self._connectors.values()]

    async def connect_all(self):
        """Initialize all configured connectors. Skip unconfigured ones."""
        for name, connector in self._connectors.items():
            if connector.is_configured():
                await connector.connect()

    async def disconnect_all(self):
        """Graceful shutdown."""
        for connector in self._connectors.values():
            await connector.disconnect()

    async def health_check_all(self) -> dict[str, bool]:
        """Run health checks on all connected connectors."""
        results = {}
        for name, connector in self._connectors.items():
            if connector.is_configured():
                results[name] = await connector.health_check()
            else:
                results[name] = None  # not configured, skip
        return results


def build_default_registry() -> ConnectorRegistry:
    """Build the registry with all known connectors. Called once at agent startup."""
    registry = ConnectorRegistry()

    # Data connectors
    registry.register("snowflake", SnowflakeConnector())
    registry.register("databricks", DatabricksConnector())
    registry.register("azure_sql", AzureSQLConnector())
    registry.register("file", FileConnector(base_dir=config.storage.base))

    # AI connectors
    registry.register("llm", LLMConnector())
    registry.register("embedder", EmbeddingConnector())
    registry.register("nli", NLIConnector())
    registry.register("reranker", RerankerConnector())

    # Sandboxes
    registry.register("python_sandbox", PythonSandbox())
    registry.register("shell_sandbox", ShellSandbox())

    return registry
```

**Startup flow**:
```
Agent starts
    ↓
registry = build_default_registry()
    ↓
await registry.connect_all()     ← only connects configured connectors
    ↓
health = await registry.health_check_all()
    ↓
Log: {"snowflake": True, "databricks": None, "llm": True, "embedder": True, ...}
    ↓
Pass registry to CEN, Response, Memory, MML
```

---

### 7.7 Design Decisions

| Decision | Rationale |
|---|---|
| Single `config.py` with singleton | Every module imports one object. No scattered `os.getenv()` calls. One place to audit all env vars. |
| `frozen=True` configs | Immutable at runtime. No accidental mutation. Config is a read-once contract. |
| `BaseConnector` interface | Adding a new data source = implement 4 methods. Skills Engine doesn't change. |
| Registry pattern | CEN initializes all connectors once. Any module can `registry.get("snowflake")`. No import spaghetti. |
| `is_configured()` check | Unconfigured connectors are silently skipped. Agent works with whatever is available. Deploy with Snowflake only? Fine. Add Databricks later? Just set env vars. |
| Sandbox as connector | Same lifecycle (connect/disconnect/health_check). Same registry. Skills Engine calls `registry.get("python_sandbox").execute(code)`. |
| AST validation for Python sandbox | Static analysis catches dangerous imports before execution. No runtime surprises. |
| Subprocess isolation for sandboxes | AI-generated code runs in a separate process with restricted PATH and env. Can't access parent process memory. |
| Output truncation (10KB) | Prevents memory bombs from malicious or buggy code. |
| Provider-agnostic LLM connector | `config.llm.provider` switches between OpenAI/Anthropic/local. Response Layer doesn't care. |

---

### 7.8 Codebase Structure

```
connectors/
├── __init__.py
├── config.py                        ← ALL env vars, credentials, paths, sandbox policy
├── base.py                          ← BaseConnector, ConnectorType, ConnectorStatus, ConnectorInfo
├── registry.py                      ← ConnectorRegistry, build_default_registry()
├── data/
│   ├── __init__.py
│   ├── snowflake.py                 ← Snowflake data warehouse connector
│   ├── databricks.py                ← Databricks lakehouse connector
│   ├── azure_sql.py                 ← Azure SQL / ODBC connector
│   ├── rest_api.py                  ← Generic REST API client
│   └── file_connector.py            ← Local file reader (CSV, JSON, JSONL, MD)
├── ai/
│   ├── __init__.py
│   ├── llm.py                       ← LLM provider (OpenAI, Anthropic, local)
│   ├── embedder.py                  ← Embedding model (all-MiniLM-L6-v2)
│   ├── nli.py                       ← NLI cross-encoder (nli-deberta-v3-small)
│   └── reranker.py                  ← Cross-encoder reranker (future)
└── sandbox/
    ├── __init__.py
    ├── base.py                      ← BaseSandbox, SandboxResult
    ├── python_sandbox.py            ← Isolated Python execution (AST validation + subprocess)
    └── shell_sandbox.py             ← Restricted shell execution (command whitelist)
```

**Updated `/datadrive/` structure**:

```
/datadrive/
├── sqlite/         (hot.db, cold.db)
├── duckdb/         (analytics.duckdb)
├── redis/          (dump.rdb)
├── kuzu/           (graph/)
├── faiss/          (meta.index, sfm.index, lfm.index, am.index, mm.index)
├── jsonl/          ({uid}/{convo_id}/messages.jsonl + summary.json)
├── md/             (*.md LFM knowledge docs)
├── yaml/           ({domain}/*.py + *.yaml, _system_generated/)
├── models/         (embedding model, NLI model, reranker model)   ← NEW
└── sandbox/
    └── tmp/        (temp scripts, auto-cleaned)                    ← NEW
```

---

## 8. Audit Layer

The universal logging fabric. Every action in the system — no matter how small — is recorded. Audit is not a feature of any module; it is the air that every module breathes. **Nothing happens without audit seeing it.**

---

### 8.1 Design Philosophy

- **Everything is logged** — LLM calls, memory reads, memory writes, sub-agent spawns, sandbox executions, data queries, governor decisions, connector health checks, config loads, errors, user messages, system events. Everything.
- **Append-only** — Audit logs are never modified or deleted. Immutable record.
- **Zero-friction** — Logging must be nearly invisible to the developer. Decorator `@audited` + direct `audit.log()` calls.
- **Queryable** — Raw logs in JSONL for durability. DuckDB external table for ad-hoc queries.
- **Separate from Analytics** — Analytics = performance metrics & eval scores (how well did we do?). Audit = factual record of what happened (who did what, when, with what result).

---

### 8.2 Audit Event Schema

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid


@dataclass
class AuditEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    component: str = ""       # "response", "memory", "control", "connector", "mml", etc.
    action: str = ""           # "llm_call", "memory_read", "sub_agent_spawn", etc.
    actor: str = ""            # "cen", "response.thinking", "mml.consolidation", "user"
    status: str = ""           # "started", "completed", "failed", "denied"
    target: str = ""           # what was acted upon: "sfm", "snowflake", "gpt-4o", etc.
    details: dict = field(default_factory=dict)  # action-specific payload
    duration_ms: float | None = None
    error: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    conversation_id: str | None = None
```

**Example events**:

```json
{"event_id": "a1b2c3", "timestamp": "2026-05-20T14:30:01Z", "component": "connector.ai", "action": "llm_call", "actor": "response.thinking", "status": "completed", "target": "gpt-4o", "details": {"tokens_in": 1200, "tokens_out": 450, "cost_usd": 0.028}, "duration_ms": 2340}

{"event_id": "d4e5f6", "timestamp": "2026-05-20T14:30:00Z", "component": "control", "action": "governor_authorize", "actor": "cen", "status": "denied", "target": "llm_call", "details": {"reason": "rate_limited", "suggestion": "wait 3s"}}

{"event_id": "g7h8i9", "timestamp": "2026-05-20T14:29:58Z", "component": "memory", "action": "memory_write", "actor": "mml.consolidation", "status": "completed", "target": "sfm", "details": {"fact_id": "f_12345", "operation": "update", "confidence_before": 0.7, "confidence_after": 0.85}}

{"event_id": "j0k1l2", "timestamp": "2026-05-20T14:29:55Z", "component": "connector.sandbox", "action": "sandbox_execute", "actor": "response.skills", "status": "failed", "target": "python_sandbox", "details": {"code_hash": "abc123"}, "error": "Timeout after 30s", "duration_ms": 30000}
```

---

### 8.3 Implementation (`audit/audit.py`)

```python
import asyncio
import json
import time
import functools
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from config import config


class AuditLogger:
    """Universal audit logger. Append-only JSONL files."""

    def __init__(self):
        self._audit_dir = config.storage.base / "audit"
        self._audit_dir.mkdir(parents=True, exist_ok=True)
        self._current_file = None
        self._current_date = None

    def _get_file(self):
        """Rotate log file daily."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._current_date:
            if self._current_file:
                self._current_file.close()
            path = self._audit_dir / f"{today}.jsonl"
            self._current_file = open(path, "a", buffering=1)  # line-buffered
            self._current_date = today
        return self._current_file

    def log(self, event: AuditEvent):
        """Write an audit event. Non-blocking, append-only."""
        f = self._get_file()
        f.write(json.dumps(event.__dict__, default=str) + "\n")

    def log_raw(
        self, component: str, action: str, actor: str, status: str,
        target: str = "", details: dict | None = None,
        duration_ms: float | None = None, error: str | None = None,
        user_id: str | None = None, session_id: str | None = None,
    ):
        """Convenience method — log without constructing AuditEvent manually."""
        event = AuditEvent(
            component=component, action=action, actor=actor, status=status,
            target=target, details=details or {}, duration_ms=duration_ms,
            error=error, user_id=user_id, session_id=session_id,
        )
        self.log(event)

    @asynccontextmanager
    async def track(self, component: str, action: str, actor: str, target: str = "", **extra):
        """Context manager that logs start + end with duration."""
        event = AuditEvent(
            component=component, action=action, actor=actor,
            status="started", target=target, details=extra,
        )
        self.log(event)
        start = time.monotonic()
        try:
            yield event
            elapsed = (time.monotonic() - start) * 1000
            event.status = "completed"
            event.duration_ms = elapsed
            self.log(event)
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            event.status = "failed"
            event.duration_ms = elapsed
            event.error = str(e)
            self.log(event)
            raise

    def close(self):
        if self._current_file:
            self._current_file.close()


# ── Singleton ──
audit = AuditLogger()


# ── Decorator ──
def audited(component: str, action: str, actor: str = ""):
    """Decorator that wraps any function with audit logging."""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            _actor = actor or func.__qualname__
            start = time.monotonic()
            audit.log_raw(component, action, _actor, "started")
            try:
                result = await func(*args, **kwargs)
                elapsed = (time.monotonic() - start) * 1000
                audit.log_raw(component, action, _actor, "completed", duration_ms=elapsed)
                return result
            except Exception as e:
                elapsed = (time.monotonic() - start) * 1000
                audit.log_raw(component, action, _actor, "failed", error=str(e), duration_ms=elapsed)
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            _actor = actor or func.__qualname__
            start = time.monotonic()
            audit.log_raw(component, action, _actor, "started")
            try:
                result = func(*args, **kwargs)
                elapsed = (time.monotonic() - start) * 1000
                audit.log_raw(component, action, _actor, "completed", duration_ms=elapsed)
                return result
            except Exception as e:
                elapsed = (time.monotonic() - start) * 1000
                audit.log_raw(component, action, _actor, "failed", error=str(e), duration_ms=elapsed)
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator
```

**Usage across the entire system**:

```python
# ── Decorator style ──

# In connectors/ai/llm.py
from audit.audit import audited

class LLMConnector:
    @audited("connector.ai", "llm_call")
    async def generate(self, prompt, system=None, **kwargs):
        # ... (audit automatically logs start, completion/failure, duration)

# In memory/management/recall.py
from audit.audit import audited

@audited("memory", "recall")
async def recall(query: str, memory_types: list[str]) -> list[MemoryResult]:
    # ...

# In control/governor.py
from audit.audit import audited

class Governor:
    @audited("control", "governor_authorize")
    async def authorize(self, action: str, **kwargs) -> GuardrailVerdict:
        # ...


# ── Context manager style (for fine-grained tracking) ──

# In response/pipeline.py
from audit.audit import audit

async def deep_pipeline(query: str):
    async with audit.track("response", "deep_pipeline", "response.pipeline", target=query[:100]):
        thinking_result = await thinking.analyze(query)
        async with audit.track("response", "recall_bridge", "response.pipeline"):
            memories = await recall_bridge.fetch(thinking_result)
        async with audit.track("response", "synthesis", "response.pipeline"):
            response = await synthesis.compose(thinking_result, memories)
    return response


# ── Direct call style (for one-off events) ──

# In control/cen.py
from audit.audit import audit

async def _handle_event(self, event):
    audit.log_raw("control", "event_received", "cen", "completed",
                  target=event.type, details={"priority": event.priority})

# In connector startup
async def connect_all(self):
    for name, connector in self._connectors.items():
        audit.log_raw("connector", "connect", "registry", "started", target=name)
        try:
            await connector.connect()
            audit.log_raw("connector", "connect", "registry", "completed", target=name)
        except Exception as e:
            audit.log_raw("connector", "connect", "registry", "failed", target=name, error=str(e))
```

---

### 8.4 Storage & Querying

**Storage**: Append-only JSONL in `/datadrive/audit/`

```
/datadrive/audit/
├── 2026-05-18.jsonl     ← one file per day, auto-rotated
├── 2026-05-19.jsonl
└── 2026-05-20.jsonl     ← today's log
```

**Querying via DuckDB** (ad-hoc, from analytics or DMN self-reflection):

```sql
-- DuckDB can read JSONL files directly — no ingestion needed
SELECT action, status, COUNT(*), AVG(duration_ms)
FROM read_json_auto('/datadrive/audit/2026-05-20.jsonl')
GROUP BY action, status
ORDER BY COUNT(*) DESC;

-- All denied actions today
SELECT *
FROM read_json_auto('/datadrive/audit/2026-05-20.jsonl')
WHERE status = 'denied';

-- Cost tracking: total LLM spend today
SELECT SUM(details->>'cost_usd' :: FLOAT) as total_spend
FROM read_json_auto('/datadrive/audit/2026-05-20.jsonl')
WHERE action = 'llm_call' AND status = 'completed';

-- Slowest operations
SELECT action, target, duration_ms
FROM read_json_auto('/datadrive/audit/2026-05-20.jsonl')
WHERE duration_ms IS NOT NULL
ORDER BY duration_ms DESC
LIMIT 20;

-- Query across multiple days
SELECT *
FROM read_json_auto('/datadrive/audit/*.jsonl')
WHERE component = 'connector.ai'
AND timestamp >= '2026-05-18';
```

---

### 8.5 Design Decisions

| Decision | Rationale |
|---|---|
| JSONL not a database | Append-only, no schema migrations, no write contention, no corruption risk. DuckDB reads JSONL directly for queries. |
| Daily file rotation | Keeps files manageable. Easy to archive/compress old logs. No log rotation daemon needed. |
| Line-buffered writes | Every event hits disk immediately. No lost events on crash. |
| Singleton `audit` | One import, one call. Zero friction. `from audit.audit import audit` everywhere. |
| `@audited` decorator | Wraps any function with start/end logging. Developer writes zero audit code for standard patterns. |
| `audit.track()` context manager | For operations with sub-steps. Captures duration and error automatically. |
| Separate from Analytics | Analytics = aggregated performance metrics (DuckDB tables, 5-min cache). Audit = raw event stream (every single thing). Different write patterns, different query patterns, different retention. |
| No filtering / log levels | Everything is logged. Period. Filtering happens at query time, not write time. Storage is cheap. Missing data is expensive. |
| `details` as free-form dict | Every action has different metadata. Schema-per-action is over-engineering. DuckDB's JSON functions handle ad-hoc queries. |

---

### 8.6 Codebase Structure

```
audit/
├── __init__.py
└── audit.py                         ← AuditLogger, AuditEvent, @audited decorator, audit singleton
```

**Updated `/datadrive/` structure**:

```
/datadrive/
├── sqlite/         (hot.db, cold.db)
├── duckdb/         (analytics.duckdb)
├── redis/          (dump.rdb)
├── kuzu/           (graph/)
├── faiss/          (meta.index, sfm.index, lfm.index, am.index, mm.index)
├── jsonl/          ({uid}/{convo_id}/messages.jsonl + summary.json)
├── md/             (*.md LFM knowledge docs)
├── yaml/           ({domain}/*.py + *.yaml, _system_generated/)
├── models/         (embedding model, NLI model, reranker model)
├── sandbox/tmp/    (temp scripts, auto-cleaned)
└── audit/          ({date}.jsonl — append-only, never deleted)          ← NEW
```

---

## 9. Agent Framework

How the agent starts, how clients connect, how it handles multiple users, and how it recovers from failures. This is the glue that turns all the layers into a running system.

---

### 9.1 Entrypoint / Bootstrap (`agent/main.py`)

**Purpose**: The `main()` that wires every system together and starts the event loop. Called once, runs forever (until shutdown signal).

**Boot sequence**:

```
python -m agent.main
    ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 1: Config                                                      │
│  Load config.py → validate env vars → fail fast if critical missing   │
│                                                                       │
│  Phase 2: Connectors                                                  │
│  Build registry → connect_all() → health_check_all() → log status    │
│                                                                       │
│  Phase 3: Memory                                                      │
│  Init SQLite (hot.db + cold.db) → Redis → Kuzu → FAISS indexes       │
│  Load ABM (frozen identity) → warm WM cache                          │
│                                                                       │
│  Phase 4: Core Systems                                                │
│  Init MML → Analytics → Response Layer → PM Scheduler                 │
│                                                                       │
│  Phase 5: Control                                                     │
│  Init Governor → Salience → DMN → CEN (wires everything)             │
│                                                                       │
│  Phase 6: Audit                                                       │
│  Init AuditLogger → log boot event                                    │
│                                                                       │
│  Phase 7: API                                                         │
│  Start WebSocket server → accept connections                          │
│                                                                       │
│  Phase 8: Run                                                         │
│  CEN.run() → event loop starts → agent is live                       │
└──────────────────────────────────────────────────────────────────────┘
```

**Implementation**:

```python
import asyncio
import signal
from connectors.config import config
from connectors.registry import build_default_registry
from memory.types import abm, wm, sfm, lfm, am, mm, em, pm, meta
from memory.management.mml import MemoryManagementLayer
from analytics.analytics import AnalyticsLayer
from response.pipeline import ResponseLayer
from control.salience import SalienceNetwork
from control.cen import CentralExecutiveNetwork
from control.dmn import DefaultModeNetwork
from control.governor import Governor, GovernorLimits
from audit.audit import audit
from agent.api import WebSocketServer
from agent.sessions import SessionManager


async def bootstrap() -> tuple["CentralExecutiveNetwork", "ConnectorRegistry"]:
    """Wire all systems and return the running CEN and registry."""

    audit.log_raw("agent", "boot", "main", "started")

    # Phase 1: Config (already loaded via singleton)
    audit.log_raw("agent", "config_loaded", "main", "completed",
                  details={"debug": config.debug})

    # Phase 2: Connectors
    registry = build_default_registry()
    await registry.connect_all()
    health = await registry.health_check_all()
    audit.log_raw("agent", "connectors_ready", "main", "completed",
                  details={"health": health})

    # Phase 3: Memory
    memory_layer = await init_memory(registry)
    audit.log_raw("agent", "memory_ready", "main", "completed")

    # Phase 4: Core Systems
    mml = MemoryManagementLayer(memory_layer, registry)
    analytics = AnalyticsLayer(config.storage.duckdb_path)
    response = ResponseLayer(memory_layer, mml, registry, analytics)
    audit.log_raw("agent", "core_ready", "main", "completed")

    # Phase 5: Control
    governor = Governor(GovernorLimits())
    salience = SalienceNetwork()
    dmn = DefaultModeNetwork(mml, analytics, memory_layer.pm)
    cen = CentralExecutiveNetwork(
        salience=salience,
        response_layer=response,
        mml=mml,
        pm_scheduler=memory_layer.pm,
        dmn=dmn,
        analytics=analytics,
        governor=governor,
    )
    audit.log_raw("agent", "control_ready", "main", "completed")

    # Phase 6: Sessions + API
    sessions = SessionManager(cen, registry)
    api = WebSocketServer(sessions, cen)
    await api.start(host=config.api.ws_host, port=config.api.ws_port)
    audit.log_raw("agent", "api_ready", "main", "completed",
                  details={"port": config.api.ws_port})

    audit.log_raw("agent", "boot", "main", "completed")
    return cen, registry


async def shutdown(cen: CentralExecutiveNetwork, registry: "ConnectorRegistry"):
    """Graceful shutdown — close everything in reverse order."""
    audit.log_raw("agent", "shutdown", "main", "started")
    cen.stop()
    await registry.disconnect_all()
    audit.close()


def main():
    loop = asyncio.new_event_loop()

    cen, registry = loop.run_until_complete(bootstrap())

    # Graceful shutdown on SIGTERM/SIGINT
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(shutdown(cen, registry)))

    try:
        loop.run_until_complete(cen.run())
    except KeyboardInterrupt:
        loop.run_until_complete(shutdown(cen, registry))
    finally:
        loop.close()


if __name__ == "__main__":
    main()
```

**Fail-fast on boot**: If critical connectors (LLM, embedder, Redis) fail health check → agent refuses to start. Optional connectors (Snowflake, Databricks) log a warning and skip.

---

### 9.2 Streaming API Layer (`agent/api.py`)

**Purpose**: WebSocket server that streams the agent's internal process to the client in real-time. Users see thinking, tool calls, memory recalls, decisions, and errors as they happen — not a black box.

**Why WebSocket**:
- Bidirectional — user sends messages AND can cancel/interrupt mid-stream
- Persistent connection — fits chat UX
- Real-time streaming — no polling

#### 9.2.1 Stream Event Schema

Every step in the pipeline emits a typed event to the connected client:

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json


@dataclass
class StreamEvent:
    type: str              # event type (see table below)
    text: str              # human-readable description for UI
    stage: str = ""        # pipeline stage: "gate" | "thinking" | "recall" | "execution" | "synthesis"
    details: dict = field(default_factory=dict)  # type-specific payload
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sequence: int = 0      # ordering guarantee within a request
    is_final: bool = False # True for the last event in a request

    def to_json(self) -> str:
        return json.dumps(self.__dict__, default=str)
```

**Event Types**:

| Type | When | UI Rendering | Example `text` |
|---|---|---|---|
| `thinking` | Agent is reasoning | Collapsible "Behind the scenes" | "Analyzing query... domain: billing, intent: revenue_leakage" |
| `memory_recall` | Searching/retrieving memory | Light text + source badge | "Searching billing knowledge base... found 3 documents" |
| `tool_call` | Calling a tool/skill | Tool name + params | "Querying Snowflake: billing_transactions WHERE month = 'March'" |
| `tool_result` | Tool returned results | Result summary | "Retrieved 1,247 records (42ms)" |
| `decision` | Decision engine chose a path | Light text | "Need Snowflake query + sandbox analysis" |
| `sub_agent` | Sub-agent spawned/completed | Agent badge | "Spawned: collections_analyzer — analyzing write-offs" |
| `governor` | Guardrail triggered | Warning badge | "Rate limit: switching to cheaper model" |
| `error` | Something failed | Red inline error | "Snowflake query timed out — retrying with cache" |
| `response` | Final response streaming | Main chat bubble (token by token) | "Based on March billing data..." |
| `done` | Request complete | Hidden (triggers UI cleanup) | — |

#### 9.2.2 Event Stream

The pipeline passes an `EventStream` object through every stage. Each stage emits events as it works:

```python
import asyncio
from collections.abc import AsyncIterator


class EventStream:
    """Bridges the pipeline to the WebSocket. Thread-safe async queue."""

    def __init__(self, session_id: str, conversation_id: str):
        self.session_id = session_id
        self.conversation_id = conversation_id
        self._queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        self._sequence = 0
        self._closed = False

    def emit(self, type: str, text: str, stage: str = "", **details):
        """Called by pipeline stages to emit an event."""
        if self._closed:
            return
        self._sequence += 1
        event = StreamEvent(
            type=type, text=text, stage=stage,
            details=details, sequence=self._sequence,
        )
        self._queue.put_nowait(event)

    def emit_thinking(self, text: str, stage: str = "thinking"):
        self.emit("thinking", text, stage=stage)

    def emit_tool_call(self, tool: str, description: str, params: dict | None = None):
        self.emit("tool_call", description, stage="execution", tool=tool, params=params or {})

    def emit_tool_result(self, tool: str, summary: str, duration_ms: float = 0):
        self.emit("tool_result", summary, stage="execution", tool=tool, duration_ms=duration_ms)

    def emit_memory(self, text: str, source: str = "", count: int = 0):
        self.emit("memory_recall", text, stage="recall", source=source, count=count)

    def emit_error(self, text: str, recoverable: bool = True):
        self.emit("error", text, stage="", recoverable=recoverable)

    def emit_response_chunk(self, text: str):
        """Stream final response token by token."""
        self.emit("response", text, stage="synthesis")

    def emit_done(self, summary: dict | None = None):
        """Signal request complete."""
        event = StreamEvent(type="done", text="", details=summary or {}, is_final=True)
        event.sequence = self._sequence + 1
        self._queue.put_nowait(event)
        self._closed = True

    async def __aiter__(self) -> AsyncIterator[StreamEvent]:
        """WebSocket handler consumes this."""
        while True:
            event = await self._queue.get()
            yield event
            if event.is_final:
                break
```

#### 9.2.3 How Pipeline Stages Emit Events

```python
# In response/gate.py
class Gate:
    async def classify(self, message: str, stream: EventStream) -> str:
        stream.emit_thinking("Classifying query type...", stage="gate")
        result = self._rule_classify(message)
        stream.emit_thinking(f"Classification: {result}", stage="gate")
        return result


# In response/thinking.py
class ThinkingEngine:
    async def analyze(self, query: str, stream: EventStream) -> ThinkingResult:
        stream.emit_thinking("Breaking down the query...")
        # LLM call to decompose
        result = await self.llm.generate(prompt, system=system_prompt)
        stream.emit_thinking(
            f"Domain: {result.domain}, Intent: {result.intent}, "
            f"Sub-questions: {len(result.sub_questions)}"
        )
        return result


# In memory/management/recall.py
async def recall(query: str, memory_types: list[str], stream: EventStream):
    stream.emit_memory(f"Searching {', '.join(memory_types)}...")

    results = []
    for mem_type in memory_types:
        hits = await search_index(mem_type, query)
        stream.emit_memory(f"Found {len(hits)} results in {mem_type}", source=mem_type, count=len(hits))
        results.extend(hits)

    return results


# In response/skills/engine.py
class SkillsEngine:
    async def execute(self, skill: Skill, params: dict, stream: EventStream):
        stream.emit_tool_call(skill.name, f"Executing: {skill.description}", params)

        start = time.monotonic()
        try:
            result = await skill.executor(params)
            elapsed = (time.monotonic() - start) * 1000
            stream.emit_tool_result(skill.name, f"Completed ({elapsed:.0f}ms)", elapsed)
            return result
        except Exception as e:
            stream.emit_error(f"{skill.name} failed: {e}", recoverable=True)
            raise


# In response/synthesis.py
class Synthesis:
    async def compose(self, context: dict, stream: EventStream) -> str:
        stream.emit_thinking("Composing final response...", stage="synthesis")

        # Stream LLM response token by token
        async for chunk in self.llm.generate_stream(prompt, system=system_prompt):
            stream.emit_response_chunk(chunk)

        stream.emit_done({
            "tools_used": context.get("tools_used", []),
            "memories_recalled": context.get("memories_recalled", 0),
            "total_duration_ms": context.get("total_duration_ms", 0),
            "llm_calls": context.get("llm_calls", 0),
            "cost_usd": context.get("cost_usd", 0),
        })
```

#### 9.2.4 WebSocket Server

```python
import asyncio
import json
import websockets
from audit.audit import audit


class WebSocketServer:
    def __init__(self, sessions: "SessionManager", cen: "CentralExecutiveNetwork"):
        self.sessions = sessions
        self.cen = cen
        self._server = None

    async def start(self, host: str = "0.0.0.0", port: int = 8765):
        self._server = await websockets.serve(self._handler, host, port)
        audit.log_raw("agent", "ws_server_started", "api", "completed",
                      details={"host": host, "port": port})

    async def _handler(self, websocket, path):
        """Handle a single WebSocket connection."""
        session = await self.sessions.create_session(websocket)
        audit.log_raw("agent", "ws_connect", "api", "completed",
                      details={"session_id": session.id, "remote": str(websocket.remote_address)})

        try:
            async for raw_message in websocket:
                message = json.loads(raw_message)
                await self._handle_message(session, websocket, message)
        except websockets.ConnectionClosed:
            pass
        finally:
            await self.sessions.end_session(session.id)
            audit.log_raw("agent", "ws_disconnect", "api", "completed",
                          details={"session_id": session.id})

    async def _handle_message(self, session, websocket, message: dict):
        """Route incoming WebSocket messages."""
        msg_type = message.get("type", "user_message")

        match msg_type:
            case "user_message":
                # Create event stream for this request
                stream = EventStream(session.id, message.get("conversation_id", "default"))

                # Start streaming events to client in background
                sender_task = asyncio.create_task(
                    self._stream_to_client(websocket, stream)
                )

                # Process through CEN (this emits events to stream as it works)
                await self.cen.handle_user_message(
                    session_id=session.id,
                    user_id=session.user_id,
                    message=message["content"],
                    conversation_id=message.get("conversation_id"),
                    stream=stream,
                )

                await sender_task  # wait for all events to be sent

            case "cancel":
                # User wants to cancel in-progress request
                await self.cen.cancel_request(session.id)
                await websocket.send(json.dumps({
                    "type": "cancelled", "text": "Request cancelled",
                }))

            case "ping":
                await websocket.send(json.dumps({"type": "pong"}))

    async def _stream_to_client(self, websocket, stream: EventStream):
        """Consume events from the stream and send to WebSocket."""
        async for event in stream:
            try:
                await websocket.send(event.to_json())
            except websockets.ConnectionClosed:
                break
```

#### 9.2.5 Client-Side Rendering

The web interface receives stream events and renders them in two tiers:

```
┌──────────────────────────────────────────────────────────────┐
│  Chat Interface                                               │
│                                                               │
│  User: What's the March revenue leakage in billing?           │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  ▼ Behind the scenes (click to expand)                  │  │
│  │                                                         │  │
│  │  💭 Analyzing query... domain: billing,                 │  │
│  │     intent: revenue_leakage, time: March                │  │
│  │  🔍 Searching billing knowledge base... 3 docs found    │  │
│  │  🔧 Querying Snowflake: billing_transactions (42ms)     │  │
│  │  📊 Retrieved 1,247 records                             │  │
│  │  🔧 Running revenue_leakage_calc.py (180ms)             │  │
│  │  💭 Composing final response...                         │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  Agent: Based on March billing data, I identified $42,300     │
│  in revenue leakage across three categories:                  │
│                                                               │
│  1. Unbilled services: $18,200 (43%)                          │
│  2. Rate discrepancies: $14,800 (35%)                         │
│  3. Credit over-applications: $9,300 (22%)                    │
│  ...                                                          │
│                                                               │
│  ⏱ 4.5s · 2 tools · 3 memories · $0.03                      │
└──────────────────────────────────────────────────────────────┘
```

**Rendering rules**:
| Event Type | Render As |
|---|---|
| `thinking` | 💭 light gray text in collapsible section |
| `memory_recall` | 🔍 with source badge |
| `tool_call` | 🔧 with tool name |
| `tool_result` | 📊 with duration |
| `decision` | 🧭 light text |
| `sub_agent` | 🤖 with agent name |
| `governor` | ⚠️ yellow warning |
| `error` | ❌ red inline (with recovery info if recoverable) |
| `response` | Main chat bubble — tokens stream in real-time |
| `done` | Footer: duration, tools used, cost |

---

### 9.3 Multi-User / Session Management (`agent/sessions.py`)

**Purpose**: Handle concurrent users with isolated sessions. Each user gets their own conversation context, WM state, and stream.

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid


@dataclass
class Session:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    conversation_id: str = ""
    websocket: object = None           # WebSocket connection
    active_stream: EventStream | None = None  # current in-progress request
    metadata: dict = field(default_factory=dict)


class SessionManager:
    def __init__(self, cen, registry):
        self.cen = cen
        self.registry = registry
        self._sessions: dict[str, Session] = {}

    async def create_session(self, websocket) -> Session:
        """Create a new session for a WebSocket connection."""
        session = Session(websocket=websocket)
        self._sessions[session.id] = session
        audit.log_raw("agent", "session_created", "sessions", "completed",
                      details={"session_id": session.id})
        return session

    async def authenticate(self, session_id: str, user_id: str, token: str) -> bool:
        """Authenticate a user within a session. Called after WS connect."""
        session = self._sessions.get(session_id)
        if not session:
            return False
        # Validate token (implementation depends on auth system)
        if await self._validate_token(user_id, token):
            session.user_id = user_id
            audit.log_raw("agent", "session_auth", "sessions", "completed",
                          details={"session_id": session_id, "user_id": user_id})
            return True
        audit.log_raw("agent", "session_auth", "sessions", "failed",
                      details={"session_id": session_id, "user_id": user_id})
        return False

    async def end_session(self, session_id: str):
        """Clean up session on disconnect."""
        session = self._sessions.pop(session_id, None)
        if session and session.active_stream:
            session.active_stream.emit_done({"reason": "session_ended"})
        audit.log_raw("agent", "session_ended", "sessions", "completed",
                      details={"session_id": session_id})

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def active_count(self) -> int:
        return len(self._sessions)

    async def _validate_token(self, user_id: str, token: str) -> bool:
        """Token validation — plug in your auth system here."""
        # Placeholder: accept all in dev mode
        if config.debug:
            return True
        # Production: validate JWT, API key, etc.
        return False
```

**Session isolation**:

| Concern | How Isolated |
|---|---|
| Conversations | Each session has its own `conversation_id` → separate WM JSONL files |
| Working Memory | Redis key prefix: `wm:{user_id}:{conversation_id}` |
| Streams | Each request creates a new `EventStream` — no cross-talk |
| Memory reads | Shared knowledge (SFM, LFM, AM) — same for all users |
| Memory writes | User-specific facts tagged with `user_id` in SFM |
| Governor limits | Per-agent (not per-user) — shared resource pool |

---

### 9.4 Error Handling Strategy

**Purpose**: Global error handling philosophy. How the agent recovers from failures without crashing, losing user context, or returning garbage.

#### 9.4.1 Error Categories & Responses

| Category | Example | Response |
|---|---|---|
| **Connector failure** | Snowflake timeout, LLM 429 | Retry with backoff → circuit breaker → fallback (cache/cheaper model) |
| **Memory failure** | FAISS index corrupt, Redis down | Degrade to available memories → log health alert → DMN schedules repair |
| **Sandbox failure** | Code timeout, OOM | Return error to user with explanation → suggest alternative approach |
| **Budget exhausted** | Daily LLM limit hit | Switch to cheaper model → if still over, queue for next day |
| **Auth failure** | Invalid token, expired session | Disconnect with clear error message |
| **Unknown error** | Unhandled exception | Catch at CEN level → log full trace → return graceful error to user |

#### 9.4.2 Retry Policy

```python
from dataclasses import dataclass


@dataclass
class RetryPolicy:
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_base: float = 2.0
    retryable_errors: frozenset = frozenset({
        "timeout", "rate_limited", "server_error", "connection_reset",
    })

    def delay_for_attempt(self, attempt: int) -> float:
        """Exponential backoff with cap."""
        delay = self.base_delay_seconds * (self.exponential_base ** attempt)
        return min(delay, self.max_delay_seconds)


# Default policies per connector type
RETRY_POLICIES = {
    "llm": RetryPolicy(max_retries=3, base_delay_seconds=1.0),
    "data": RetryPolicy(max_retries=2, base_delay_seconds=2.0),
    "sandbox": RetryPolicy(max_retries=1, base_delay_seconds=0),  # sandboxes: fail fast
    "memory": RetryPolicy(max_retries=2, base_delay_seconds=0.5),
}
```

#### 9.4.3 Fallback Chains

When the primary path fails, the agent falls through a chain of alternatives:

```python
FALLBACK_CHAINS = {
    "llm_call": [
        "retry_same_model",          # retry with backoff
        "switch_to_cheap_model",     # gpt-4o → gpt-4o-mini
        "use_cached_response",       # if similar query was answered before
        "return_partial_response",   # return what we have so far
        "return_error_to_user",      # "I'm unable to process this right now"
    ],
    "data_query": [
        "retry_query",               # retry with backoff
        "use_cached_data",           # last successful query result
        "try_alternative_connector", # Snowflake down → try Databricks
        "return_partial_response",   # answer with whatever data we have
        "return_error_to_user",
    ],
    "memory_recall": [
        "retry_recall",              # retry
        "skip_failed_memory_type",   # SFM failed → still use LFM, AM results
        "broaden_search",            # relax similarity threshold
        "return_without_memory",     # LLM answers from general knowledge (flagged)
    ],
}
```

#### 9.4.4 Error Streaming

Errors are streamed to the client in real-time so users see what went wrong:

```python
# In the pipeline, when a tool fails:
stream.emit_error("Snowflake query timed out — retrying with cached data", recoverable=True)

# When fallback succeeds:
stream.emit_thinking("Using cached billing data from yesterday (Snowflake unavailable)")

# When nothing works:
stream.emit_error("Unable to retrieve billing data. Please try again later.", recoverable=False)
stream.emit_done({"status": "partial_failure", "error": "data_unavailable"})
```

The user sees the error inline in the "Behind the scenes" section. No black box — they know exactly what happened and why.

---

### 9.5 Data Ingestion API (`agent/ingestion.py`)

**Purpose**: REST interface for pushing domain data into the agent's memory. Admins, pipelines, or external systems POST plain text or files → agent reads, chunks, embeds, and stores in LFM and SFM.

**Why REST (not WebSocket)**: Ingestion is fire-and-forget. No streaming needed. Standard HTTP POST with status response.

#### 9.5.1 Endpoints

| Method | Path | Content-Type | What |
|---|---|---|---|
| `POST` | `/ingest/text` | `application/json` | Ingest plain text → LFM + SFM extraction |
| `POST` | `/ingest/file` | `multipart/form-data` | Upload file (.md, .txt, .csv, .pdf) → parse → LFM + SFM |
| `POST` | `/ingest/facts` | `application/json` | Directly insert facts → SFM |
| `GET` | `/ingest/status/{job_id}` | — | Check ingestion job status |
| `GET` | `/ingest/health` | — | Health check for ingestion service |

#### 9.5.2 Implementation

```python
from aiohttp import web
from dataclasses import dataclass
from audit.audit import audit
from logger import log
from redact import redact


@dataclass
class IngestionJob:
    job_id: str
    status: str       # "queued" | "processing" | "completed" | "failed"
    source: str       # "text" | "file" | "facts"
    items_created: int = 0
    errors: list = None


class IngestionAPI:
    def __init__(self, mml, registry, embedder):
        self.mml = mml
        self.registry = registry
        self.embedder = embedder
        self._jobs: dict[str, IngestionJob] = {}

    def routes(self) -> list[web.RouteDef]:
        return [
            web.post("/ingest/text", self.ingest_text),
            web.post("/ingest/file", self.ingest_file),
            web.post("/ingest/facts", self.ingest_facts),
            web.get("/ingest/status/{job_id}", self.get_status),
            web.get("/ingest/health", self.health),
        ]

    async def ingest_text(self, request: web.Request) -> web.Response:
        """Ingest plain text → chunk → embed → store in LFM + extract facts to SFM."""
        body = await request.json()
        text = body.get("text", "")
        title = body.get("title", "untitled")
        domain = body.get("domain", "general")
        source = body.get("source", "api_upload")

        if not text:
            return web.json_response({"error": "text is required"}, status=400)

        audit.log_raw("ingestion", "ingest_text", "api", "started",
                      details={"title": title, "domain": domain, "chars": len(text)})
        log.info(f"Ingestion: text '{title}' ({len(text)} chars), domain={domain}")

        job_id = str(uuid.uuid4())
        job = IngestionJob(job_id=job_id, status="processing", source="text")
        self._jobs[job_id] = job

        try:
            # 1. Store full document in LFM (cold.db + /datadrive/md/)
            doc_id = await self.mml.lfm_ingest(
                text=text, title=title, domain=domain, source=source,
            )

            # 2. Chunk → embed → FAISS index
            chunks = self._chunk_text(text, chunk_size=512, overlap=50)
            embeddings = self.embedder.embed_batch([c["text"] for c in chunks])
            await self.mml.lfm_index_chunks(doc_id, chunks, embeddings)

            # 3. Extract facts → SFM (LLM extracts key facts from text)
            facts = await self.mml.extract_facts(text, domain=domain)
            for fact in facts:
                await self.mml.sfm_write(fact)

            job.status = "completed"
            job.items_created = 1 + len(chunks) + len(facts)  # 1 doc + N chunks + M facts

            audit.log_raw("ingestion", "ingest_text", "api", "completed",
                          details={"doc_id": doc_id, "chunks": len(chunks), "facts": len(facts)})

            return web.json_response({
                "job_id": job_id, "status": "completed",
                "doc_id": doc_id, "chunks_created": len(chunks),
                "facts_extracted": len(facts),
            })

        except Exception as e:
            job.status = "failed"
            job.errors = [str(e)]
            audit.log_raw("ingestion", "ingest_text", "api", "failed", error=str(e))
            log.error(f"Ingestion failed: {e}", exc_info=True)
            return web.json_response({"error": str(e), "job_id": job_id}, status=500)

    async def ingest_file(self, request: web.Request) -> web.Response:
        """Upload a file → parse → ingest to LFM + SFM."""
        reader = await request.multipart()
        field = await reader.next()

        if field is None or field.name != "file":
            return web.json_response({"error": "file field required"}, status=400)

        filename = field.filename
        content = await field.read(decode=True)

        # Parse based on file type
        ext = Path(filename).suffix.lower()
        match ext:
            case ".md" | ".txt":
                text = content.decode("utf-8")
            case ".csv":
                text = self._csv_to_text(content)
            case ".json":
                text = self._json_to_text(content)
            case _:
                return web.json_response(
                    {"error": f"Unsupported file type: {ext}. Supported: .md, .txt, .csv, .json"},
                    status=400,
                )

        domain = request.query.get("domain", "general")
        audit.log_raw("ingestion", "ingest_file", "api", "started",
                      details={"filename": filename, "size": len(content), "domain": domain})
        log.info(f"Ingestion: file '{filename}' ({len(content)} bytes), domain={domain}")

        # Reuse text ingestion pipeline
        fake_request = type("Req", (), {"json": lambda s: {
            "text": text, "title": filename, "domain": domain, "source": "file_upload",
        }})()
        return await self.ingest_text(fake_request)

    async def ingest_facts(self, request: web.Request) -> web.Response:
        """Directly insert facts into SFM. For structured data imports."""
        body = await request.json()
        facts = body.get("facts", [])
        domain = body.get("domain", "general")

        if not facts:
            return web.json_response({"error": "facts array is required"}, status=400)

        audit.log_raw("ingestion", "ingest_facts", "api", "started",
                      details={"count": len(facts), "domain": domain})
        log.info(f"Ingestion: {len(facts)} direct facts, domain={domain}")

        created = 0
        errors = []
        for fact in facts:
            try:
                await self.mml.sfm_write({
                    "subject": fact.get("subject", ""),
                    "predicate": fact.get("predicate", ""),
                    "object": fact.get("object", ""),
                    "domain": domain,
                    "source": "direct_ingestion",
                    "confidence": fact.get("confidence", 0.9),
                })
                created += 1
            except Exception as e:
                errors.append({"fact": fact, "error": str(e)})

        audit.log_raw("ingestion", "ingest_facts", "api", "completed",
                      details={"created": created, "errors": len(errors)})

        return web.json_response({
            "created": created, "errors": len(errors),
            "error_details": errors if errors else None,
        })

    async def get_status(self, request: web.Request) -> web.Response:
        job_id = request.match_info["job_id"]
        job = self._jobs.get(job_id)
        if not job:
            return web.json_response({"error": "Job not found"}, status=404)
        return web.json_response(job.__dict__)

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> list[dict]:
        """Split text into overlapping chunks for embedding."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunks.append({
                "text": " ".join(chunk_words),
                "start_idx": i,
                "end_idx": min(i + chunk_size, len(words)),
            })
        return chunks
```

**Ingestion flow**:

```
POST /ingest/text  {"text": "...", "title": "March Billing Report", "domain": "billing"}
    ↓
┌────────────────────────────────────────────────────────────┐
│  1. Store full document → cold.db (long_form_documents)    │
│  2. Save markdown → /datadrive/md/march_billing_report.md  │
│  3. Chunk text (512 words, 50 overlap)                     │
│  4. Embed chunks → FAISS lfm.index                         │
│  5. Store chunks → cold.db (lfm_chunks)                    │
│  6. LLM extracts key facts from text                       │
│  7. Store facts → hot.db (short_form_facts)                │
│  8. Embed facts → FAISS sfm.index                          │
│  9. Create AM graph edges (entities from facts)             │
└────────────────────────────────────────────────────────────┘
    ↓
200 OK {"doc_id": "...", "chunks_created": 12, "facts_extracted": 8}
```

---

### 9.6 Design Decisions

| Decision | Rationale |
|---|---|
| WebSocket over REST/SSE | Bidirectional: user can send messages AND cancel mid-stream. Persistent connection fits chat UX. |
| Streaming every pipeline stage | Transparency builds trust. Users see the agent working, not a spinning loader. Debug-friendly: errors visible inline. |
| Two-tier rendering (behind-the-scenes + response) | Thinking/tools are noise for most users → collapsible. Final response is the main output → prominent. |
| `EventStream` as async queue | Decouples pipeline from transport. Pipeline emits events without knowing if client is WebSocket, SSE, or test harness. |
| Session isolation via key prefixes | Simple, no separate DBs per user. Redis `wm:{uid}:{cid}`, JSONL `{uid}/{cid}/`. Shared read-only knowledge. |
| Fail-fast boot for critical connectors | No point running if LLM or embedder is down. Optional connectors (Snowflake) can be added later. |
| Fallback chains over hard failures | Agent should degrade gracefully: cheaper model → cached data → partial response → error. Never crash. |
| Exponential backoff with cap | Standard retry pattern. Prevents thundering herd on rate limits. Cap prevents absurd delays. |
| Governor limits shared across users | Simpler. Total system capacity is what matters, not per-user slicing. Per-user limits can be added later if needed. |
| Auth as pluggable `_validate_token` | No opinion on auth system. JWT, API key, OAuth — plug in what you use. Dev mode accepts all. |
| REST for ingestion, WebSocket for chat | Ingestion is fire-and-forget (POST → 200). Chat needs bidirectional streaming. Right tool for each job. |
| LLM-based fact extraction on ingest | Automatically populates SFM from uploaded documents. No manual fact entry needed. |
| Chunking with overlap | 50-word overlap prevents losing context at chunk boundaries. Standard RAG practice. |

---

### 9.7 Codebase Structure

```
agent/
├── __init__.py
├── main.py                          ← Entrypoint: bootstrap + main()
├── api.py                           ← WebSocket server + message routing
├── ingestion.py                     ← REST API for data ingestion (POST text/files → LFM + SFM)
├── sessions.py                      ← Session management + user isolation
├── stream.py                        ← EventStream + StreamEvent schema
└── errors.py                        ← RetryPolicy, fallback chains, error categories
```

---

---

## 10. Cross-Cutting Utilities

Root-level modules used by every layer. These are the plumbing — config, logging, PII redaction, and prompt management.

---

### 10.1 Root Configuration (`config.py`)

**Purpose**: Single source of truth for ALL environment variables. Promoted from `connectors/config.py` to root because every layer depends on it — not just connectors.

**Location**: `CogniCore/config.py` (root level)

**Import everywhere**: `from config import config`

The full `AgentConfig` dataclass (with `StorageConfig`, `RedisConfig`, `LLMConfig`, `EmbeddingConfig`, `NLIConfig`, `SandboxConfig`, `SnowflakeConfig`, `DatabricksConfig`, `AzureSQLConfig`) remains as documented in Section 7.1 — just moved to root.

**Additional configs for new utilities**:

```python
@dataclass(frozen=True)
class LoggingConfig:
    level: str = os.getenv("LOG_LEVEL", "INFO")        # DEBUG | INFO | WARNING | ERROR
    log_dir: str = os.getenv("LOG_DIR", "/datadrive/logs")
    max_file_size_mb: int = int(os.getenv("LOG_MAX_SIZE_MB", "50"))
    backup_count: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    log_to_console: bool = os.getenv("LOG_CONSOLE", "true").lower() == "true"


@dataclass(frozen=True)
class RedactionConfig:
    enabled: bool = os.getenv("REDACTION_ENABLED", "true").lower() == "true"
    redact_ssn: bool = True
    redact_credit_card: bool = True
    redact_phone: bool = True
    redact_email: bool = True
    redact_account_numbers: bool = True
    custom_patterns: list = None     # additional regex patterns from config file
    replacement: str = "[REDACTED]"


@dataclass(frozen=True)
class APIConfig:
    ws_host: str = os.getenv("WS_HOST", "0.0.0.0")
    ws_port: int = int(os.getenv("WS_PORT", "8765"))
    ingestion_host: str = os.getenv("INGESTION_HOST", "0.0.0.0")
    ingestion_port: int = int(os.getenv("INGESTION_PORT", "8766"))
    cors_origins: str = os.getenv("CORS_ORIGINS", "*")


@dataclass(frozen=True)
class AgentConfig:
    # ... (all existing configs from Section 7.1) ...
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    redaction: RedactionConfig = field(default_factory=RedactionConfig)
    api: APIConfig = field(default_factory=APIConfig)


config = AgentConfig()
```

---

### 10.2 Logger (`logger.py`)

**Purpose**: Standard Python logging. DEBUG, INFO, WARNING, ERROR. For developers and ops. Not the same as Audit (which logs every action as structured events) or Stream Events (which go to the user).

**The three observability layers**:

| Layer | What | Who | Where |
|---|---|---|---|
| **Logger** | DEBUG/INFO/WARN/ERROR | Developers, ops | Console + `/datadrive/logs/agent.log` |
| **Stream Events** | Pipeline steps in real-time | End users (WebSocket) | Chat UI "Behind the scenes" |
| **Audit** | Every action with status/duration | Compliance, analytics, DMN | `/datadrive/audit/{date}.jsonl` |

**Implementation**:

```python
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from config import config


def setup_logger() -> logging.Logger:
    """Create the agent-wide logger. Called once at import time."""
    logger = logging.getLogger("agent")
    logger.setLevel(getattr(logging, config.logging.level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(name)s.%(module)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if config.logging.log_to_console:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        logger.addHandler(console)

    # Rotating file handler
    log_dir = Path(config.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "agent.log",
        maxBytes=config.logging.max_file_size_mb * 1_000_000,
        backupCount=config.logging.backup_count,
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


# ── Singleton ──
log = setup_logger()
```

**Usage**:

```python
from logger import log

# In any module:
log.debug("FAISS search returned 12 results in 8ms")
log.info("Boot complete: 7 connectors healthy, 2 skipped")
log.warning("Redis memory at 78% — approaching threshold")
log.error("Snowflake connection failed", exc_info=True)

# Child loggers for module-specific filtering:
log_recall = log.getChild("recall")
log_recall.debug("Hybrid search: FAISS=8, BM25=5, graph=3 → merged=12")

log_cen = log.getChild("cen")
log_cen.info("Mode switch: active → idle (5min timeout)")
```

**Log output**:

```
2026-05-20 14:30:01 | INFO  | agent.main    | Boot complete: 7 connectors healthy, 2 skipped
2026-05-20 14:30:02 | DEBUG | agent.recall  | Hybrid search: FAISS=8, BM25=5, graph=3 → merged=12
2026-05-20 14:30:05 | WARN  | agent.cen     | Redis memory at 78% — approaching threshold
2026-05-20 14:30:12 | ERROR | agent.skills  | Snowflake connection failed
```

---

### 10.3 PII Redaction (`redact.py`)

**Purpose**: Scrub sensitive data before it hits logs, audit, or stream events. some data contains SSNs, account numbers, credit cards, phone numbers — none of that should appear in plain text in logs.

**Applied at three points**:
1. **Logger** — `log.info(redact(message))` for any message that might contain user data
2. **Audit** — `audit.log()` passes `details` through redaction before writing
3. **Stream Events** — `stream.emit()` redacts `text` before sending to client

**Implementation**:

```python
import re
from config import config


class Redactor:
    """PII redaction engine. Regex-based pattern matching."""

    # Standard PII patterns for  domain
    PATTERNS = {
        "ssn": {
            "regex": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "label": "[SSN]",
            "enabled_key": "redact_ssn",
        },
        "credit_card": {
            "regex": re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),
            "label": "[CREDIT_CARD]",
            "enabled_key": "redact_credit_card",
        },
        "phone": {
            "regex": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
            "label": "[PHONE]",
            "enabled_key": "redact_phone",
        },
        "email": {
            "regex": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "label": "[EMAIL]",
            "enabled_key": "redact_email",
        },
        "account_number": {
            "regex": re.compile(r"\b(?:acct?|account)[#:\s]*\d{6,12}\b", re.IGNORECASE),
            "label": "[ACCOUNT]",
            "enabled_key": "redact_account_numbers",
        },
    }

    def __init__(self):
        self._active_patterns = []
        if config.redaction.enabled:
            for name, pattern in self.PATTERNS.items():
                if getattr(config.redaction, pattern["enabled_key"], True):
                    self._active_patterns.append(pattern)

    def redact(self, text: str) -> str:
        """Redact all PII from text. Returns cleaned string."""
        if not config.redaction.enabled or not text:
            return text
        for pattern in self._active_patterns:
            text = pattern["regex"].sub(pattern["label"], text)
        return text

    def redact_dict(self, data: dict) -> dict:
        """Deep-redact all string values in a dict."""
        if not config.redaction.enabled:
            return data
        cleaned = {}
        for key, value in data.items():
            if isinstance(value, str):
                cleaned[key] = self.redact(value)
            elif isinstance(value, dict):
                cleaned[key] = self.redact_dict(value)
            elif isinstance(value, list):
                cleaned[key] = [self.redact(v) if isinstance(v, str) else v for v in value]
            else:
                cleaned[key] = value
        return cleaned


# ── Singleton ──
_redactor = Redactor()
redact = _redactor.redact
redact_dict = _redactor.redact_dict
```

**Usage**:

```python
from redact import redact, redact_dict

# In logger calls:
log.info(redact(f"Processing request for account {account_num}"))
# Output: "Processing request for account [ACCOUNT]"

# In audit:
audit.log_raw("memory", "sfm_write", "mml", "completed",
              details=redact_dict({"fact": "Customer SSN is 123-45-6789"}))
# Stored as: {"fact": "Customer SSN is [SSN]"}

# In stream events:
stream.emit_thinking(redact(f"Found customer record: {raw_data}"))
# Client sees: "Found customer record: name=John, ssn=[SSN], phone=[PHONE]"
```

**What gets redacted**:

| Pattern | Input | Output |
|---|---|---|
| SSN | `123-45-6789` | `[SSN]` |
| Credit card | `4111 1111 1111 1111` | `[CREDIT_CARD]` |
| Phone | `(555) 123-4567` | `[PHONE]` |
| Email | `john@example.com` | `[EMAIL]` |
| Account | `acct# 123456789` | `[ACCOUNT]` |

---

### 10.4 Prompt Management (`prompts/`)

**Purpose**: All LLM system prompts stored as `.md` files. Easy to version, review, edit, and A/B test without touching Python code.

**Structure**:

```
prompts/
├── system/                          ← System prompts (injected as system message)
│   ├── identity.md                  ← ABM identity prompt (who am I, what can I do)
│   ├── thinking.md                  ← Deep Pipeline: query decomposition
│   ├── synthesis.md                 ← Deep Pipeline: response composition
│   ├── decision.md                  ← Deep Pipeline: action planning
│   ├── creativity.md                ← Deep Pipeline: novel insights
│   ├── prediction.md                ← Deep Pipeline: forecasting
│   └── gate.md                      ← Gate: trivial query classification (if SLM)
│
├── extraction/                      ← Prompts for data extraction tasks
│   ├── fact_extraction.md           ← Extract SFM facts from raw text
│   ├── entity_extraction.md         ← Extract AM graph entities/relations
│   └── summary.md                   ← Summarize conversations for WM
│
├── skills/                          ← Prompts for skill execution
│   ├── code_generation.md           ← Generate Python code for sandbox
│   ├── sql_generation.md            ← Generate SQL for data connectors
│   └── analysis.md                  ← Data analysis prompt
│
└── meta/                            ← Prompts for meta/self-reflection
    ├── self_reflection.md           ← DMN: assess own performance
    ├── conflict_resolution.md       ← MML: resolve contradicting facts
    └── learning.md                  ← MML: generalization, abstraction
```

**Prompt loading**:

```python
from pathlib import Path
from config import config


class PromptManager:
    """Load and cache prompts from .md files."""

    def __init__(self, prompts_dir: str | Path = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._cache: dict[str, str] = {}

    def get(self, path: str, **variables) -> str:
        """Load a prompt by path (relative to prompts/). Supports variable substitution.

        Usage: prompt_mgr.get("system/thinking.md", domain="billing", query=user_query)
        """
        if path not in self._cache:
            full_path = self.prompts_dir / path
            if not full_path.exists():
                raise FileNotFoundError(f"Prompt not found: {full_path}")
            self._cache[path] = full_path.read_text()

        prompt = self._cache[path]

        # Variable substitution: {{variable_name}} → value
        for key, value in variables.items():
            prompt = prompt.replace(f"{{{{{key}}}}}", str(value))

        return prompt

    def reload(self, path: str | None = None):
        """Clear cache. Useful for hot-reloading prompts during development."""
        if path:
            self._cache.pop(path, None)
        else:
            self._cache.clear()


# ── Singleton ──
prompts = PromptManager()
```

**Example prompt file** (`prompts/system/thinking.md`):

```markdown
You are a query decomposition engine for a finance domain AI agent.

Given a user query, break it down into:
1. **Domain**: Which area? (billing, collections, revenue, general)
2. **Intent**: What does the user want? (lookup, analysis, comparison, forecast, explanation)
3. **Entities**: Key entities mentioned (account IDs, dates, dollar amounts, product names)
4. **Sub-questions**: Break complex queries into atomic sub-questions
5. **Data sources needed**: Which data sources are required? (SFM facts, LFM docs, Snowflake, etc.)

Domain context: {{domain}}
User query: {{query}}
Conversation history: {{history}}

Respond in JSON format.
```

**Usage in pipeline**:

```python
from prompts import prompts

# In response/thinking.py
class ThinkingEngine:
    async def analyze(self, query: str, domain: str, history: str) -> ThinkingResult:
        system_prompt = prompts.get("system/thinking.md",
            domain=domain, query=query, history=history,
        )
        result = await self.llm.generate(query, system=system_prompt)
        return parse_thinking_result(result)

# In memory/management/mml.py
class MML:
    async def extract_facts(self, text: str, domain: str) -> list[dict]:
        prompt = prompts.get("extraction/fact_extraction.md",
            text=text, domain=domain,
        )
        result = await self.llm.generate(prompt)
        return parse_facts(result)
```

**Why `.md` files**:
| Reason | Detail |
|---|---|
| Version control | Git diff shows exactly what changed in a prompt |
| Non-developer editing | Domain experts can edit prompts without touching Python |
| A/B testing | Swap prompt files to test different approaches |
| Hot-reload | `prompts.reload()` in dev mode — no restart needed |
| Separation of concerns | Business logic (Python) separate from instructions (prompts) |

---

### 10.5 Design Decisions

| Decision | Rationale |
|---|---|
| `config.py` at root, not in connectors | Config is used by every layer. Root-level = shortest import path, clearest ownership. |
| `logger.py` as Python `logging` wrapper | Standard library, zero dependencies. Rotating files for production. Child loggers for module filtering. |
| Redaction as regex patterns | Fast (<1ms), no ML model needed. Finance PII patterns are well-defined (SSN, CC, phone). Configurable per-pattern enable/disable. |
| Redaction applied at write time | Scrub before data hits disk or network. Once redacted, PII is gone — no accidental leaks in logs/audit/streams. |
| Prompts as `.md` files | Git-friendly, editable by non-developers, hot-reloadable. Variable substitution with `{{var}}` keeps prompts dynamic. |
| Prompt cache with manual reload | Prompts rarely change at runtime. Cache for performance. `reload()` for dev iteration. |
| Three separate observability layers | Logger (dev debug) ≠ Stream (user transparency) ≠ Audit (compliance record). Different audiences, different formats, different retention. |

---

### 10.6 Codebase Structure

```
CogniCore/                  ← Root
├── config.py                        ← ALL env vars, credentials, paths — ONE place (root level)
├── logger.py                        ← Python logging (DEBUG/INFO/WARN/ERROR)
├── redact.py                        ← PII redaction (SSN, CC, phone, email, account)
└── prompts/                         ← LLM prompt templates (.md files)
    ├── system/                      ← System prompts (identity, thinking, synthesis, etc.)
    ├── extraction/                  ← Data extraction prompts (facts, entities, summaries)
    ├── skills/                      ← Skill execution prompts (code gen, SQL gen)
    └── meta/                        ← Meta prompts (self-reflection, conflict resolution)
```

---

## 11. Testing Strategy

Comprehensive testing across all layers. Tests are first-class citizens — not an afterthought.

---

### 11.1 Test Structure

```
tests/
├── unit/                            ← Fast, isolated tests (no external deps)
│   ├── memory/
│   │   ├── test_abm.py              ← Autobiographical Memory
│   │   ├── test_wm.py               ← Working Memory (mock Redis)
│   │   ├── test_sfm.py              ← Short Form Memory
│   │   ├── test_lfm.py              ← Long Form Memory
│   │   ├── test_am.py               ← Associative Memory (mock Kuzu)
│   │   ├── test_mm.py               ← Motor Memory
│   │   ├── test_em.py               ← Emotional Memory
│   │   ├── test_pm.py               ← Prospective Memory
│   │   └── test_meta.py             ← Meta Memory
│   ├── control/
│   │   ├── test_salience.py         ← Priority scoring
│   │   ├── test_cen.py              ← Event routing
│   │   ├── test_dmn.py              ← Self-reflection
│   │   └── test_governor.py         ← Guardrails, rate limits
│   ├── response/
│   │   ├── test_gate.py             ← Query classification
│   │   ├── test_thinking.py         ← Query decomposition
│   │   ├── test_decision.py         ← Execution planning
│   │   └── test_synthesis.py        ← Response composition
│   ├── connectors/
│   │   ├── test_llm.py              ← LLM connector (mock API)
│   │   ├── test_embedder.py         ← Embedding connector
│   │   └── test_sandbox.py          ← Sandbox validation
│   └── utils/
│       ├── test_redact.py           ← PII redaction
│       └── test_prompts.py          ← Prompt loading
│
├── integration/                     ← Tests with real dependencies
│   ├── test_memory_flow.py          ← Write → recall → verify
│   ├── test_recall_pipeline.py      ← Full hybrid search
│   ├── test_learning_cycle.py       ← Consolidation → learning → verification
│   ├── test_connector_health.py     ← All connectors health check
│   └── test_ingestion_flow.py       ← Ingest → chunk → embed → store
│
├── e2e/                             ← End-to-end scenarios
│   ├── test_chat_flow.py            ← User message → response
│   ├── test_multi_turn.py           ← Conversation with context
│   ├── test_scheduled_tasks.py      ← PM task execution
│   └── test_error_recovery.py       ← Failure → fallback → recovery
│
├── fixtures/                        ← Test data
│   ├── prompts/                     ← Test prompt templates
│   ├── documents/                   ← Sample LFM documents
│   ├── facts/                       ← Sample SFM facts
│   └── conversations/               ← Sample WM conversations
│
└── conftest.py                      ← Pytest fixtures, mocks
```

---

### 11.2 Testing Patterns

**Unit Tests**: Mock all external dependencies. Test logic in isolation.

```python
# tests/unit/memory/test_sfm.py
import pytest
from unittest.mock import Mock, AsyncMock
from memory.types.sfm import ShortFormMemory


@pytest.fixture
def mock_db():
    """Mock SQLite connection."""
    db = Mock()
    db.execute = Mock(return_value=Mock(fetchall=Mock(return_value=[])))
    return db


@pytest.fixture
def mock_embedder():
    """Mock embedding connector."""
    embedder = AsyncMock()
    embedder.embed_async = AsyncMock(return_value=np.zeros(1024))
    return embedder


@pytest.mark.asyncio
async def test_sfm_write(mock_db, mock_embedder):
    sfm = ShortFormMemory(db=mock_db, embedder=mock_embedder)
    
    fact = {
        "subject": "Customer 123",
        "predicate": "has_status",
        "object": "active",
        "domain": "billing",
        "confidence": 0.9,
    }
    
    fact_id = await sfm.write(fact)
    
    assert fact_id is not None
    mock_db.execute.assert_called()
    mock_embedder.embed_async.assert_called_once()


@pytest.mark.asyncio
async def test_sfm_recall(mock_db, mock_embedder):
    sfm = ShortFormMemory(db=mock_db, embedder=mock_embedder)
    
    results = await sfm.recall("customer status", top_k=5)
    
    assert isinstance(results, list)
```

**Integration Tests**: Use real databases with test data.

```python
# tests/integration/test_memory_flow.py
import pytest
from memory.management.mml import MemoryManagementLayer


@pytest.fixture
async def mml(test_registry):
    """MML with real SQLite (in-memory) and FAISS."""
    return MemoryManagementLayer(test_registry)


@pytest.mark.asyncio
async def test_write_then_recall(mml):
    # Write a fact
    fact = {"subject": "Test", "predicate": "is", "object": "working", "domain": "test"}
    await mml.sfm_write(fact)
    
    # Recall it
    results = await mml.recall("test working", memory_types=["sfm"])
    
    assert len(results) >= 1
    assert any("working" in r.content for r in results)
```

**E2E Tests**: Full pipeline with mocked LLM responses.

```python
# tests/e2e/test_chat_flow.py
import pytest
from agent.api import handle_message


@pytest.fixture
def mock_llm_responses():
    """Predefined LLM responses for deterministic testing."""
    return {
        "thinking": '{"domain": "billing", "intent": "lookup"}',
        "synthesis": "Based on the data, the customer status is active.",
    }


@pytest.mark.asyncio
async def test_simple_query(test_agent, mock_llm_responses):
    response = await test_agent.handle("What is customer 123's status?")
    
    assert response.status == "completed"
    assert "customer" in response.text.lower()
    assert response.tools_used == []  # Simple lookup, no tools
```

---

### 11.3 Test Fixtures

```python
# tests/conftest.py
import pytest
import tempfile
from pathlib import Path
from config import AgentConfig


@pytest.fixture
def test_config():
    """Config with temp directories for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig(
            storage=StorageConfig(base=Path(tmpdir)),
            redis=RedisConfig(host="localhost", port=6379, db=15),  # test DB
            llm=LLMConfig(provider="mock"),
            debug=True,
        )
        yield config


@pytest.fixture
async def test_registry(test_config):
    """Connector registry with mock connectors."""
    from connectors.registry import ConnectorRegistry
    from tests.mocks import MockLLMConnector, MockEmbeddingConnector
    
    registry = ConnectorRegistry()
    registry.register("llm", MockLLMConnector())
    registry.register("embedder", MockEmbeddingConnector())
    await registry.connect_all()
    yield registry
    await registry.disconnect_all()


@pytest.fixture
def sample_facts():
    """Sample SFM facts for testing."""
    return [
        {"subject": "Customer 123", "predicate": "has_balance", "object": "$500", "domain": "billing"},
        {"subject": "Customer 123", "predicate": "has_status", "object": "active", "domain": "billing"},
        {"subject": "Product A", "predicate": "has_price", "object": "$99", "domain": "catalog"},
    ]


@pytest.fixture
def sample_document():
    """Sample LFM document for testing."""
    return {
        "title": "Billing Policy",
        "content": "All invoices are due within 30 days. Late payments incur a 5% fee.",
        "domain": "billing",
    }
```

---

### 11.4 Mock Connectors

```python
# tests/mocks.py
from connectors.base import BaseConnector, ConnectorInfo, ConnectorType, ConnectorStatus
import numpy as np


class MockLLMConnector(BaseConnector):
    """Mock LLM for deterministic testing."""
    
    def __init__(self, responses: dict | None = None):
        self.responses = responses or {}
        self.call_log = []
    
    async def connect(self): pass
    async def disconnect(self): pass
    async def health_check(self) -> bool: return True
    
    async def generate(self, prompt: str, system: str | None = None, **kwargs) -> str:
        self.call_log.append({"prompt": prompt, "system": system})
        # Return predefined response or generic
        for key, response in self.responses.items():
            if key in prompt.lower():
                return response
        return '{"result": "mock response"}'
    
    def info(self) -> ConnectorInfo:
        return ConnectorInfo("mock_llm", ConnectorType.AI, ConnectorStatus.CONNECTED, {})


class MockEmbeddingConnector(BaseConnector):
    """Mock embedder with deterministic vectors."""
    
    def __init__(self, dimension: int = 1024):
        self._dimension = dimension
    
    async def connect(self): pass
    async def disconnect(self): pass
    async def health_check(self) -> bool: return True
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    async def embed_async(self, text: str) -> np.ndarray:
        # Deterministic embedding based on text hash
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(self._dimension).astype(np.float32)
    
    async def embed_batch_async(self, texts: list[str]) -> np.ndarray:
        return np.array([await self.embed_async(t) for t in texts])
    
    def info(self) -> ConnectorInfo:
        return ConnectorInfo("mock_embedder", ConnectorType.AI, ConnectorStatus.CONNECTED, {})
```

---

### 11.5 Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only (fast)
pytest tests/unit/ -v --tb=short

# Integration tests (requires Redis, SQLite)
pytest tests/integration/ -v

# E2E tests
pytest tests/e2e/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html

# Specific test file
pytest tests/unit/memory/test_sfm.py -v

# Run tests matching pattern
pytest tests/ -k "recall" -v
```

---

### 11.6 CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements-dev.txt
      - run: pytest tests/unit/ -v --tb=short

  integration-tests:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7
        ports:
          - 6379:6379
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements-dev.txt
      - run: pytest tests/integration/ -v
```

---

## 12. Evaluation System

Dedicated module for monitoring learning effectiveness, memory quality, and agent performance. Separate from Analytics (which tracks operational metrics) — Evals tracks **quality**.

---

### 12.1 Purpose

- **Learning Effectiveness**: Are the 10 learning types actually improving the agent?
- **Memory Quality**: Are facts accurate? Are procedures working? Are associations useful?
- **Recall Quality**: Is the right information being retrieved?
- **Response Quality**: Are users satisfied? Are answers correct?

---

### 12.2 Eval Types

| Eval Type | What It Measures | Frequency | Storage |
|-----------|------------------|-----------|---------|
| **Fact Accuracy** | Are SFM facts still true? | Daily | DuckDB |
| **Procedure Success** | Do MM procedures complete without errors? | Per execution | DuckDB |
| **Recall Precision** | Is retrieved content relevant to query? | Per recall | DuckDB |
| **Recall Coverage** | Did we miss relevant content? | Sampled | DuckDB |
| **Response Satisfaction** | User feedback (thumbs up/down) | Per response | DuckDB |
| **Learning Drift** | Are learned facts diverging from ground truth? | Weekly | DuckDB |
| **Coherence Score** | Do SFM facts contradict each other? | Daily | DuckDB |
| **Knowledge Gap Closure** | Are identified gaps being filled? | Weekly | DuckDB |

---

### 12.3 Implementation (`evals/`)

```
evals/
├── __init__.py
├── runner.py                        ← Eval orchestrator
├── metrics.py                       ← Metric definitions and calculations
├── datasets.py                      ← Ground truth datasets
├── reporters.py                     ← Report generation
└── types/
    ├── fact_accuracy.py             ← SFM fact verification
    ├── procedure_success.py         ← MM procedure tracking
    ├── recall_quality.py            ← Recall precision/coverage
    ├── response_quality.py          ← User satisfaction
    ├── learning_drift.py            ← Learning effectiveness
    └── coherence.py                 ← Internal consistency
```

---

### 12.4 Eval Runner

```python
# evals/runner.py
import asyncio
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Callable
from analytics.analytics import AnalyticsLayer
from audit.audit import audit


@dataclass
class EvalResult:
    eval_type: str
    score: float              # 0.0 - 1.0
    sample_size: int
    passed: int
    failed: int
    details: dict
    timestamp: str = None
    
    def __post_init__(self):
        self.timestamp = self.timestamp or datetime.now(timezone.utc).isoformat()


class EvalRunner:
    """Orchestrates all evaluation types."""
    
    def __init__(self, mml, analytics: AnalyticsLayer):
        self.mml = mml
        self.analytics = analytics
        self._evals: dict[str, Callable] = {}
    
    def register(self, name: str, eval_fn: Callable):
        """Register an eval function."""
        self._evals[name] = eval_fn
    
    async def run_all(self) -> list[EvalResult]:
        """Run all registered evals."""
        results = []
        for name, eval_fn in self._evals.items():
            audit.log_raw("evals", "eval_started", "runner", "started", target=name)
            try:
                result = await eval_fn(self.mml)
                results.append(result)
                self._store_result(result)
                audit.log_raw("evals", "eval_completed", "runner", "completed",
                              target=name, details={"score": result.score})
            except Exception as e:
                audit.log_raw("evals", "eval_failed", "runner", "failed",
                              target=name, error=str(e))
        return results
    
    async def run_single(self, name: str) -> EvalResult:
        """Run a specific eval."""
        if name not in self._evals:
            raise ValueError(f"Unknown eval: {name}")
        return await self._evals[name](self.mml)
    
    def _store_result(self, result: EvalResult):
        """Store eval result in DuckDB."""
        self.analytics.log_eval(
            eval_type=result.eval_type,
            score=result.score,
            sample_size=result.sample_size,
            passed=result.passed,
            failed=result.failed,
            details=result.details,
        )
```

---

### 12.5 Fact Accuracy Eval

```python
# evals/types/fact_accuracy.py
from evals.runner import EvalResult


async def eval_fact_accuracy(mml) -> EvalResult:
    """Verify SFM facts against ground truth or source documents."""
    
    # Get sample of facts to verify
    facts = await mml.sfm_sample(n=100, strategy="random")
    
    passed = 0
    failed = 0
    failures = []
    
    for fact in facts:
        # Check if fact has a source document
        if fact.source_doc_id:
            # Verify against source
            source = await mml.lfm_get(fact.source_doc_id)
            is_valid = await verify_fact_against_source(fact, source, mml.nli)
        else:
            # No source — check for contradictions with other facts
            is_valid = not await mml.has_contradictions(fact)
        
        if is_valid:
            passed += 1
        else:
            failed += 1
            failures.append({"fact_id": fact.id, "reason": "contradiction_or_drift"})
    
    return EvalResult(
        eval_type="fact_accuracy",
        score=passed / len(facts) if facts else 1.0,
        sample_size=len(facts),
        passed=passed,
        failed=failed,
        details={"failures": failures[:10]},  # top 10 failures
    )


async def verify_fact_against_source(fact, source, nli) -> bool:
    """Use NLI to check if fact is entailed by source."""
    fact_text = f"{fact.subject} {fact.predicate} {fact.object}"
    result = nli.predict(source.content[:1000], fact_text)
    return result["entailment"] > 0.7
```

---

### 12.6 Recall Quality Eval

```python
# evals/types/recall_quality.py
from evals.runner import EvalResult


async def eval_recall_quality(mml) -> EvalResult:
    """Evaluate recall precision using labeled query-document pairs."""
    
    # Load test queries with known relevant documents
    test_set = load_recall_test_set()  # From evals/datasets/
    
    precision_scores = []
    
    for query, relevant_ids in test_set:
        # Run recall
        results = await mml.recall(query, memory_types=["sfm", "lfm"], top_k=10)
        retrieved_ids = {r.id for r in results}
        
        # Calculate precision@10
        relevant_retrieved = len(retrieved_ids & set(relevant_ids))
        precision = relevant_retrieved / len(results) if results else 0
        precision_scores.append(precision)
    
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    
    return EvalResult(
        eval_type="recall_precision",
        score=avg_precision,
        sample_size=len(test_set),
        passed=sum(1 for p in precision_scores if p >= 0.5),
        failed=sum(1 for p in precision_scores if p < 0.5),
        details={"precision_distribution": precision_scores},
    )
```

---

### 12.7 DuckDB Eval Schema

```sql
-- Eval results table (in analytics.duckdb)
CREATE TABLE IF NOT EXISTS eval_results (
    id TEXT PRIMARY KEY,
    eval_type TEXT NOT NULL,
    score DOUBLE NOT NULL,
    sample_size INTEGER,
    passed INTEGER,
    failed INTEGER,
    details JSON,
    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Eval trends view
CREATE VIEW eval_trends AS
SELECT 
    eval_type,
    DATE_TRUNC('day', ts) as day,
    AVG(score) as avg_score,
    MIN(score) as min_score,
    MAX(score) as max_score,
    COUNT(*) as eval_count
FROM eval_results
GROUP BY eval_type, DATE_TRUNC('day', ts)
ORDER BY day DESC;
```

---

### 12.8 Scheduled Evals

Evals are scheduled via PM (Prospective Memory):

```python
# In agent/main.py bootstrap
async def schedule_evals(pm, eval_runner):
    """Schedule recurring evals."""
    
    # Daily evals
    await pm.create_schedule({
        "task_name": "daily_fact_accuracy",
        "task_type": "eval",
        "cron_expr": "0 3 * * *",  # 3 AM daily
        "payload": {"eval": "fact_accuracy"},
    })
    
    await pm.create_schedule({
        "task_name": "daily_coherence",
        "task_type": "eval",
        "cron_expr": "0 4 * * *",  # 4 AM daily
        "payload": {"eval": "coherence"},
    })
    
    # Weekly evals
    await pm.create_schedule({
        "task_name": "weekly_learning_drift",
        "task_type": "eval",
        "cron_expr": "0 5 * * 0",  # 5 AM Sunday
        "payload": {"eval": "learning_drift"},
    })
```

---

## 13. Versioning & Migration

Schema versioning, FAISS index migration, and procedure versioning strategies.

---

### 13.1 SQLite Schema Versioning

```python
# migrations/sqlite_migrations.py
from pathlib import Path
import sqlite3


MIGRATIONS = [
    # Version 1: Initial schema
    (1, """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO schema_version (version) VALUES (1);
    """),
    
    # Version 2: Add confidence to facts
    (2, """
        ALTER TABLE short_form_facts ADD COLUMN confidence REAL DEFAULT 0.8;
        UPDATE schema_version SET version = 2;
    """),
    
    # Version 3: Add source tracking
    (3, """
        ALTER TABLE short_form_facts ADD COLUMN source_doc_id TEXT;
        ALTER TABLE short_form_facts ADD COLUMN source_chunk_idx INTEGER;
        UPDATE schema_version SET version = 3;
    """),
]


def get_current_version(conn: sqlite3.Connection) -> int:
    """Get current schema version."""
    try:
        cur = conn.execute("SELECT MAX(version) FROM schema_version")
        return cur.fetchone()[0] or 0
    except sqlite3.OperationalError:
        return 0


def migrate(db_path: Path):
    """Apply pending migrations."""
    conn = sqlite3.connect(db_path)
    current = get_current_version(conn)
    
    for version, sql in MIGRATIONS:
        if version > current:
            print(f"Applying migration {version}...")
            conn.executescript(sql)
            conn.commit()
    
    conn.close()
```

---

### 13.2 FAISS Index Versioning

When embedding model changes, all indexes must be rebuilt.

```python
# migrations/faiss_migrations.py
from pathlib import Path
import json
from config import config


def get_index_metadata(index_path: Path) -> dict:
    """Read index metadata (embedding model, dimension, count)."""
    meta_path = index_path.with_suffix(".meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def save_index_metadata(index_path: Path, metadata: dict):
    """Save index metadata."""
    meta_path = index_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2))


async def check_index_compatibility(index_path: Path, embedder) -> bool:
    """Check if index is compatible with current embedding model."""
    meta = get_index_metadata(index_path)
    
    if not meta:
        return False  # No metadata = needs rebuild
    
    return (
        meta.get("model") == config.embedding.model_name and
        meta.get("dimension") == embedder.dimension
    )


async def rebuild_index(index_name: str, mml, embedder):
    """Rebuild a FAISS index with current embedding model."""
    from audit.audit import audit
    
    audit.log_raw("migration", "index_rebuild", "faiss", "started", target=index_name)
    
    index_path = config.storage.faiss_dir / f"{index_name}.index"
    
    # Get all records that need re-embedding
    if index_name == "sfm":
        records = await mml.sfm_get_all()
        texts = [f"{r.subject} {r.predicate} {r.object}" for r in records]
        ids = [r.id for r in records]
    elif index_name == "lfm":
        chunks = await mml.lfm_get_all_chunks()
        texts = [c.text for c in chunks]
        ids = [c.id for c in chunks]
    # ... other index types
    
    # Re-embed all
    embeddings = await embedder.embed_batch_async(texts)
    
    # Rebuild index
    import faiss
    index = faiss.IndexFlatIP(embedder.dimension)
    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    
    # Save metadata
    save_index_metadata(index_path, {
        "model": config.embedding.model_name,
        "dimension": embedder.dimension,
        "count": len(ids),
        "rebuilt_at": datetime.now(timezone.utc).isoformat(),
    })
    
    audit.log_raw("migration", "index_rebuild", "faiss", "completed",
                  target=index_name, details={"count": len(ids)})
```

---

### 13.3 Procedure Versioning

Motor Memory procedures have explicit version tracking.

```python
# In memory/types/mm.py

@dataclass
class Procedure:
    id: str
    name: str
    version: int              # Incremented on each update
    code_path: Path
    yaml_path: Path
    created_at: str
    updated_at: str
    deprecated: bool = False
    superseded_by: str | None = None  # ID of newer version


async def update_procedure(self, proc_id: str, new_code: str, new_yaml: dict):
    """Update a procedure, creating a new version."""
    old = await self.get(proc_id)
    
    # Create new version
    new_version = old.version + 1
    new_id = f"{old.name}_v{new_version}"
    
    # Write new files
    new_code_path = self._write_code(new_id, new_code)
    new_yaml_path = self._write_yaml(new_id, new_yaml)
    
    # Mark old as deprecated
    await self._deprecate(proc_id, superseded_by=new_id)
    
    # Register new
    return await self._register(new_id, new_code_path, new_yaml_path, version=new_version)
```

---

### 13.4 Migration on Startup

```python
# In agent/main.py

async def run_migrations(registry):
    """Run all pending migrations on startup."""
    from migrations.sqlite_migrations import migrate as migrate_sqlite
    from migrations.faiss_migrations import check_index_compatibility, rebuild_index
    
    # SQLite migrations
    migrate_sqlite(config.storage.hot_db)
    migrate_sqlite(config.storage.cold_db)
    
    # FAISS index compatibility check
    embedder = registry.get("embedder")
    for index_name in ["sfm", "lfm", "am", "mm", "meta"]:
        index_path = config.storage.faiss_dir / f"{index_name}.index"
        if index_path.exists():
            if not await check_index_compatibility(index_path, embedder):
                log.warning(f"Index {index_name} incompatible with current embedder, rebuilding...")
                await rebuild_index(index_name, mml, embedder)
```

---

## 14. Working Memory Redis Fallback

Graceful degradation when Redis is unavailable.

---

### 14.1 Fallback Strategy

```python
# memory/types/wm.py
import asyncio
from pathlib import Path
from config import config
from logger import log


class WorkingMemory:
    """Working Memory with Redis primary and local file fallback."""
    
    def __init__(self, redis_client, fallback_dir: Path):
        self._redis = redis_client
        self._fallback_dir = fallback_dir
        self._fallback_mode = False
        self._local_cache: dict[str, dict] = {}
    
    async def connect(self):
        """Connect to Redis, fall back to local if unavailable."""
        try:
            await self._redis.ping()
            self._fallback_mode = False
            log.info("WM: Connected to Redis")
        except Exception as e:
            log.warning(f"WM: Redis unavailable ({e}), using local fallback")
            self._fallback_mode = True
            self._load_local_cache()
    
    def _load_local_cache(self):
        """Load cached conversations from local JSONL files."""
        for jsonl_file in self._fallback_dir.glob("**/*.jsonl"):
            key = str(jsonl_file.relative_to(self._fallback_dir))
            self._local_cache[key] = self._read_jsonl(jsonl_file)
    
    async def get_context(self, user_id: str, conversation_id: str) -> list[dict]:
        """Get conversation context."""
        key = f"wm:{user_id}:{conversation_id}"
        
        if self._fallback_mode:
            return self._local_cache.get(key, [])
        
        try:
            data = await self._redis.get(key)
            return json.loads(data) if data else []
        except Exception as e:
            log.warning(f"WM: Redis read failed ({e}), using local")
            return self._local_cache.get(key, [])
    
    async def append_message(self, user_id: str, conversation_id: str, message: dict):
        """Append a message to conversation."""
        key = f"wm:{user_id}:{conversation_id}"
        
        # Always write to local (durability)
        self._append_local(user_id, conversation_id, message)
        
        if self._fallback_mode:
            # Update local cache
            if key not in self._local_cache:
                self._local_cache[key] = []
            self._local_cache[key].append(message)
            return
        
        try:
            # Append to Redis list
            await self._redis.rpush(key, json.dumps(message))
            await self._redis.expire(key, config.redis.ttl_inactivity_hours * 3600)
        except Exception as e:
            log.warning(f"WM: Redis write failed ({e}), local only")
            self._fallback_mode = True
    
    def _append_local(self, user_id: str, conversation_id: str, message: dict):
        """Append to local JSONL file."""
        path = self._fallback_dir / user_id / conversation_id / "messages.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(message) + "\n")
    
    async def health_check(self) -> dict:
        """Return WM health status."""
        try:
            await self._redis.ping()
            redis_ok = True
        except:
            redis_ok = False
        
        return {
            "redis_connected": redis_ok,
            "fallback_mode": self._fallback_mode,
            "local_conversations": len(self._local_cache),
        }
    
    async def try_reconnect(self):
        """Periodically try to reconnect to Redis."""
        if not self._fallback_mode:
            return
        
        try:
            await self._redis.ping()
            log.info("WM: Redis reconnected, syncing local cache...")
            await self._sync_local_to_redis()
            self._fallback_mode = False
        except:
            pass  # Still unavailable
    
    async def _sync_local_to_redis(self):
        """Sync local cache back to Redis after reconnection."""
        for key, messages in self._local_cache.items():
            for msg in messages:
                await self._redis.rpush(key, json.dumps(msg))
```

---

### 14.2 Health Monitoring

```python
# In control/cen.py

async def _monitor_wm_health(self):
    """Background task to monitor WM and attempt reconnection."""
    while True:
        await asyncio.sleep(60)  # Check every minute
        
        health = await self.wm.health_check()
        
        if health["fallback_mode"]:
            self.analytics.log_metric("wm", "fallback_mode", 1)
            await self.wm.try_reconnect()
        else:
            self.analytics.log_metric("wm", "fallback_mode", 0)
```

---

## 15. Ingestion Rate Limiting

Rate limiting for the ingestion REST API.

---

### 15.1 Governor Integration

```python
# In control/governor.py — add to GovernorLimits

@dataclass
class GovernorLimits:
    # ... existing limits ...
    
    # ── Ingestion ──
    ingestion_requests_per_minute: int = 10
    ingestion_requests_per_hour: int = 100
    ingestion_max_text_size_kb: int = 1024      # 1MB max per text
    ingestion_max_file_size_mb: int = 10        # 10MB max per file
    ingestion_max_facts_per_request: int = 100


# Add to Governor class

def _check_ingestion(self, size_kb: int = 0) -> GuardrailVerdict:
    now = time.time()
    
    # Rate limit
    ts = self._call_timestamps["ingestion"]
    self._prune_timestamps(ts, now)
    
    calls_last_minute = sum(1 for t in ts if now - t < 60)
    if calls_last_minute >= self.limits.ingestion_requests_per_minute:
        return GuardrailVerdict(
            allowed=False, reason=DenialReason.RATE_LIMITED,
            message=f"Ingestion rate limit: {calls_last_minute}/{self.limits.ingestion_requests_per_minute} per minute",
            suggestion="Wait before submitting more data",
            wait_seconds=60 - (now - ts[-self.limits.ingestion_requests_per_minute]),
        )
    
    calls_last_hour = sum(1 for t in ts if now - t < 3600)
    if calls_last_hour >= self.limits.ingestion_requests_per_hour:
        return GuardrailVerdict(
            allowed=False, reason=DenialReason.RATE_LIMITED,
            message=f"Ingestion hourly limit: {calls_last_hour}/{self.limits.ingestion_requests_per_hour}",
            suggestion="Wait before submitting more data",
        )
    
    # Size limit
    if size_kb > self.limits.ingestion_max_text_size_kb:
        return GuardrailVerdict(
            allowed=False, reason=DenialReason.SESSION_LIMIT,
            message=f"Text too large: {size_kb}KB > {self.limits.ingestion_max_text_size_kb}KB",
            suggestion="Split into smaller chunks",
        )
    
    return GuardrailVerdict(allowed=True)
```

---

### 15.2 Ingestion API Integration

```python
# In agent/ingestion.py

class IngestionAPI:
    def __init__(self, mml, registry, embedder, governor: Governor):
        self.mml = mml
        self.registry = registry
        self.embedder = embedder
        self.governor = governor  # Add governor
        self._jobs: dict[str, IngestionJob] = {}

    async def ingest_text(self, request: web.Request) -> web.Response:
        body = await request.json()
        text = body.get("text", "")
        
        # Check rate limit and size
        size_kb = len(text.encode()) / 1024
        verdict = await self.governor.authorize("ingestion", size_kb=size_kb)
        
        if not verdict.allowed:
            return web.json_response({
                "error": verdict.message,
                "suggestion": verdict.suggestion,
                "wait_seconds": verdict.wait_seconds,
            }, status=429 if verdict.reason == DenialReason.RATE_LIMITED else 400)
        
        # Record the action
        self.governor.record_action("ingestion")
        
        # ... rest of ingestion logic ...
```

---

## 16. Cron & Timezone Handling

Standard library cron parsing with UTC normalization.

---

### 16.1 Implementation

```python
# memory/types/pm.py
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import re


class CronParser:
    """Simple cron expression parser using standard library.
    
    Supports: minute hour day month weekday
    Examples: "0 3 * * *" (3 AM daily), "*/15 * * * *" (every 15 min)
    """
    
    FIELDS = ["minute", "hour", "day", "month", "weekday"]
    
    def __init__(self, expr: str, tz: str = "UTC"):
        self.expr = expr
        self.tz = ZoneInfo(tz)
        self.parts = self._parse(expr)
    
    def _parse(self, expr: str) -> dict:
        parts = expr.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {expr}")
        
        return {
            field: self._parse_field(part, field)
            for field, part in zip(self.FIELDS, parts)
        }
    
    def _parse_field(self, part: str, field: str) -> set[int]:
        """Parse a single cron field into a set of valid values."""
        ranges = {
            "minute": (0, 59),
            "hour": (0, 23),
            "day": (1, 31),
            "month": (1, 12),
            "weekday": (0, 6),
        }
        min_val, max_val = ranges[field]
        
        if part == "*":
            return set(range(min_val, max_val + 1))
        
        if part.startswith("*/"):
            step = int(part[2:])
            return set(range(min_val, max_val + 1, step))
        
        if "-" in part:
            start, end = map(int, part.split("-"))
            return set(range(start, end + 1))
        
        if "," in part:
            return set(map(int, part.split(",")))
        
        return {int(part)}
    
    def next_run(self, after: datetime | None = None) -> datetime:
        """Calculate next run time after given datetime (in UTC)."""
        if after is None:
            after = datetime.now(timezone.utc)
        
        # Convert to target timezone for matching
        local = after.astimezone(self.tz)
        
        # Start from next minute
        candidate = local.replace(second=0, microsecond=0)
        candidate = candidate + timedelta(minutes=1)
        
        # Find next matching time (max 1 year search)
        for _ in range(525600):  # minutes in a year
            if self._matches(candidate):
                # Convert back to UTC
                return candidate.astimezone(timezone.utc)
            candidate += timedelta(minutes=1)
        
        raise ValueError(f"No valid run time found for {self.expr}")
    
    def _matches(self, dt: datetime) -> bool:
        """Check if datetime matches cron expression."""
        return (
            dt.minute in self.parts["minute"] and
            dt.hour in self.parts["hour"] and
            dt.day in self.parts["day"] and
            dt.month in self.parts["month"] and
            dt.weekday() in self.parts["weekday"]
        )
```

---

### 16.2 Missed Job Recovery

```python
# memory/types/pm.py

class ProspectiveMemoryScheduler:
    
    async def recover_missed_jobs(self):
        """Check for and execute missed scheduled jobs.
        
        Called daily by DMN or on startup.
        """
        from audit.audit import audit
        
        # Find jobs that were scheduled but not executed
        missed = await self._db.execute("""
            SELECT s.id, s.task_name, s.scheduled_at, s.payload
            FROM schedules s
            LEFT JOIN execution_log e ON s.id = e.schedule_id 
                AND DATE(e.started_at) = DATE(s.scheduled_at)
            WHERE s.status = 'SCHEDULED'
            AND s.scheduled_at < datetime('now')
            AND e.id IS NULL
            ORDER BY s.scheduled_at
        """).fetchall()
        
        audit.log_raw("pm", "missed_job_check", "scheduler", "completed",
                      details={"missed_count": len(missed)})
        
        for job in missed:
            # Check if job is still relevant (not too old)
            scheduled_at = datetime.fromisoformat(job["scheduled_at"])
            age_hours = (datetime.now(timezone.utc) - scheduled_at).total_seconds() / 3600
            
            if age_hours > 24:
                # Too old, mark as skipped
                await self._mark_skipped(job["id"], reason="too_old")
                continue
            
            # Execute the missed job
            audit.log_raw("pm", "missed_job_execute", "scheduler", "started",
                          target=job["task_name"])
            try:
                await self.execute_task(job)
                audit.log_raw("pm", "missed_job_execute", "scheduler", "completed",
                              target=job["task_name"])
            except Exception as e:
                audit.log_raw("pm", "missed_job_execute", "scheduler", "failed",
                              target=job["task_name"], error=str(e))
```

---

## 17. Sub-Agent Memory Access

Sub-agents have read-only access to memory via MML.

---

### 17.1 Implementation

```python
# response/sub_agents.py

@dataclass
class SubAgentConfig:
    task: str
    memory_access: bool = True     # Can read memory
    memory_write: bool = False     # Cannot write (read-only)
    max_depth: int = 1             # No nested sub-agents
    timeout_seconds: int = 30


class SubAgent:
    """Parallel worker with read-only memory access."""
    
    def __init__(self, config: SubAgentConfig, mml, parent_context: dict):
        self.config = config
        self.mml = mml
        self.parent_context = parent_context  # Shared WM context from parent
        self._read_only_mml = ReadOnlyMMLWrapper(mml)
    
    async def run(self) -> dict:
        """Execute sub-agent task."""
        from audit.audit import audit
        
        audit.log_raw("sub_agent", "started", f"sub_agent_{self.config.task[:20]}",
                      "started", details={"task": self.config.task})
        
        try:
            # Sub-agent can recall but not write
            if self.config.memory_access:
                context = await self._read_only_mml.recall(
                    self.config.task, 
                    memory_types=["sfm", "lfm", "am"]
                )
            else:
                context = []
            
            # Execute task with LLM
            result = await self._execute_with_context(context)
            
            audit.log_raw("sub_agent", "completed", f"sub_agent_{self.config.task[:20]}",
                          "completed")
            return result
            
        except Exception as e:
            audit.log_raw("sub_agent", "failed", f"sub_agent_{self.config.task[:20]}",
                          "failed", error=str(e))
            raise


class ReadOnlyMMLWrapper:
    """Wrapper that only exposes read operations."""
    
    def __init__(self, mml):
        self._mml = mml
    
    async def recall(self, query: str, memory_types: list[str], top_k: int = 10):
        """Read-only recall."""
        return await self._mml.recall(query, memory_types, top_k)
    
    async def sfm_get(self, fact_id: str):
        """Read a specific fact."""
        return await self._mml.sfm_get(fact_id)
    
    async def lfm_get(self, doc_id: str):
        """Read a specific document."""
        return await self._mml.lfm_get(doc_id)
    
    # No write methods exposed
    # sfm_write, lfm_ingest, etc. are NOT available
```

---

## 18. Token-Based Chunking

Standard token-based chunking for LFM documents.

---

### 18.1 Implementation

```python
# memory/management/chunking.py
import tiktoken


class TokenChunker:
    """Token-based document chunking with overlap.
    
    Uses tiktoken for accurate token counting (same as OpenAI models).
    """
    
    def __init__(
        self,
        chunk_size: int = 512,      # tokens per chunk
        overlap: int = 50,           # overlap tokens
        model: str = "cl100k_base",  # tiktoken encoding
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoder = tiktoken.get_encoding(model)
    
    def chunk(self, text: str) -> list[dict]:
        """Split text into overlapping token-based chunks."""
        tokens = self.encoder.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)
            
            chunks.append({
                "text": chunk_text,
                "start_token": start,
                "end_token": end,
                "token_count": len(chunk_tokens),
            })
            
            # Move start with overlap
            start = end - self.overlap
            if start >= len(tokens):
                break
        
        return chunks
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoder.encode(text))


# Usage in ingestion
async def ingest_document(text: str, title: str, domain: str):
    chunker = TokenChunker(chunk_size=512, overlap=50)
    chunks = chunker.chunk(text)
    
    # Embed and store chunks
    for i, chunk in enumerate(chunks):
        embedding = await embedder.embed_async(chunk["text"])
        await store_chunk(doc_id, i, chunk, embedding)
```

---

## 19. Batched Memory Writes

Reduce I/O amplification with write batching.

---

### 19.1 Write Queue

```python
# memory/management/write_queue.py
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable


@dataclass
class WriteOperation:
    memory_type: str           # "sfm", "lfm", "am", etc.
    operation: str             # "insert", "update", "delete"
    data: dict
    callback: Callable | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class WriteQueue:
    """Batches memory writes to reduce I/O operations.
    
    Flushes on:
    - Batch size reached (default: 50)
    - Time elapsed (default: 5 seconds)
    - Explicit flush call
    """
    
    def __init__(
        self,
        mml,
        batch_size: int = 50,
        flush_interval_seconds: float = 5.0,
    ):
        self.mml = mml
        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds
        self._queue: list[WriteOperation] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None
    
    async def start(self):
        """Start the background flush task."""
        self._flush_task = asyncio.create_task(self._flush_loop())
    
    async def stop(self):
        """Stop and flush remaining writes."""
        if self._flush_task:
            self._flush_task.cancel()
        await self.flush()
    
    async def enqueue(self, op: WriteOperation):
        """Add a write operation to the queue."""
        async with self._lock:
            self._queue.append(op)
            
            if len(self._queue) >= self.batch_size:
                await self._flush_batch()
    
    async def flush(self):
        """Force flush all pending writes."""
        async with self._lock:
            await self._flush_batch()
    
    async def _flush_loop(self):
        """Background task that flushes periodically."""
        while True:
            await asyncio.sleep(self.flush_interval)
            async with self._lock:
                if self._queue:
                    await self._flush_batch()
    
    async def _flush_batch(self):
        """Flush current batch to storage."""
        if not self._queue:
            return
        
        batch = self._queue.copy()
        self._queue.clear()
        
        # Group by memory type for efficient bulk operations
        by_type: dict[str, list[WriteOperation]] = {}
        for op in batch:
            by_type.setdefault(op.memory_type, []).append(op)
        
        # Execute bulk writes
        for mem_type, ops in by_type.items():
            await self._bulk_write(mem_type, ops)
        
        # Call callbacks
        for op in batch:
            if op.callback:
                try:
                    await op.callback()
                except Exception:
                    pass  # Don't fail batch on callback error
    
    async def _bulk_write(self, mem_type: str, ops: list[WriteOperation]):
        """Execute bulk write for a memory type."""
        from audit.audit import audit
        
        audit.log_raw("memory", "bulk_write", "write_queue", "started",
                      target=mem_type, details={"count": len(ops)})
        
        if mem_type == "sfm":
            await self.mml.sfm_bulk_write([op.data for op in ops])
        elif mem_type == "meta":
            await self.mml.meta_bulk_register([op.data for op in ops])
        # ... other types
        
        audit.log_raw("memory", "bulk_write", "write_queue", "completed",
                      target=mem_type, details={"count": len(ops)})
```

---

### 19.2 Usage

```python
# In MML initialization
class MemoryManagementLayer:
    def __init__(self, memory_layer, registry):
        self.write_queue = WriteQueue(self, batch_size=50, flush_interval_seconds=5.0)
    
    async def start(self):
        await self.write_queue.start()
    
    async def sfm_write_queued(self, fact: dict):
        """Queue a fact write instead of immediate write."""
        await self.write_queue.enqueue(WriteOperation(
            memory_type="sfm",
            operation="insert",
            data=fact,
        ))
```

---

## 20. FAISS Auditing

All FAISS operations are audited.

---

### 20.1 Audited FAISS Wrapper

```python
# memory/management/faiss_wrapper.py
import faiss
import numpy as np
from pathlib import Path
from audit.audit import audit


class AuditedFAISSIndex:
    """FAISS index wrapper with full audit logging."""
    
    def __init__(self, name: str, dimension: int, index_path: Path):
        self.name = name
        self.dimension = dimension
        self.index_path = index_path
        self._index: faiss.Index | None = None
        self._id_map: dict[int, str] = {}  # FAISS ID → external ID
    
    async def load(self):
        """Load index from disk."""
        if self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            self._load_id_map()
            audit.log_raw("faiss", "load", "faiss_wrapper", "completed",
                          target=self.name, details={"count": self._index.ntotal})
        else:
            self._index = faiss.IndexFlatIP(self.dimension)
            audit.log_raw("faiss", "create", "faiss_wrapper", "completed",
                          target=self.name, details={"dimension": self.dimension})
    
    async def add(self, external_id: str, embedding: np.ndarray):
        """Add a single embedding."""
        faiss_id = self._index.ntotal
        self._index.add(embedding.reshape(1, -1))
        self._id_map[faiss_id] = external_id
        
        audit.log_raw("faiss", "add", "faiss_wrapper", "completed",
                      target=self.name, details={"external_id": external_id, "faiss_id": faiss_id})
    
    async def add_batch(self, external_ids: list[str], embeddings: np.ndarray):
        """Add multiple embeddings."""
        start_id = self._index.ntotal
        self._index.add(embeddings)
        
        for i, ext_id in enumerate(external_ids):
            self._id_map[start_id + i] = ext_id
        
        audit.log_raw("faiss", "add_batch", "faiss_wrapper", "completed",
                      target=self.name, details={"count": len(external_ids)})
    
    async def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """Search for similar embeddings."""
        distances, indices = self._index.search(query_embedding.reshape(1, -1), top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx in self._id_map:
                results.append((self._id_map[idx], float(dist)))
        
        audit.log_raw("faiss", "search", "faiss_wrapper", "completed",
                      target=self.name, details={"top_k": top_k, "results": len(results)})
        
        return results
    
    async def save(self):
        """Save index to disk."""
        faiss.write_index(self._index, str(self.index_path))
        self._save_id_map()
        
        audit.log_raw("faiss", "save", "faiss_wrapper", "completed",
                      target=self.name, details={"count": self._index.ntotal})
    
    def _load_id_map(self):
        map_path = self.index_path.with_suffix(".idmap.json")
        if map_path.exists():
            import json
            self._id_map = {int(k): v for k, v in json.loads(map_path.read_text()).items()}
    
    def _save_id_map(self):
        import json
        map_path = self.index_path.with_suffix(".idmap.json")
        map_path.write_text(json.dumps(self._id_map))
```

---

### 20.2 Architecture Completeness

| Concern | Covered? | Where |
|---|---|---|
| Input handling | ✅ | WebSocket (chat) + REST (ingestion) |
| Processing / reasoning | ✅ | Response Layer (gate → deep pipeline → synthesis) |
| Memory (read + write) | ✅ | 9 memory types + MML + recall |
| Intelligence | ✅ | AI connectors (LLM, embedder, NLI) |
| Data access | ✅ | Data connectors (Snowflake, Databricks, Azure SQL, REST, File) |
| Code execution | ✅ | Sandboxes (Python, Shell) |
| Orchestration | ✅ | CEN (event loop) |
| Attention / priority | ✅ | Salience Network |
| Background work | ✅ | DMN + MML sleep tasks + PM scheduler |
| Safety / guardrails | ✅ | Governor (rate limits, circuit breakers, human gates, ingestion limits) |
| Security (code) | ✅ | Sandbox AST validation, blocked imports, PII redaction |
| Auth | ✅ (external) | Nginx reverse proxy + OAuth (external to agent) |
| Observability | ✅ | Logger + Audit + Stream Events + Analytics + FAISS Auditing (5 layers) |
| Configuration | ✅ | Root `config.py` (single source of truth) |
| Error recovery | ✅ | Retry policies + fallback chains + Redis fallback |
| User transparency | ✅ | Streaming pipeline events to client |
| Prompt management | ✅ | `prompts/` as .md files |
| Learning / improvement | ✅ | 10 learning types in MML |
| Self-reflection | ✅ | DMN reviews own performance |
| User feedback | ✅ | Periodic feedback prompts → EM + MML corrective learning |
| Data ingestion | ✅ | REST API (POST text, files, facts → LFM + SFM) with rate limiting |
| Testing | ✅ | Unit, integration, E2E tests (Section 11) |
| Evaluations | ✅ | Fact accuracy, recall quality, learning drift (Section 12) |
| Versioning / Migration | ✅ | SQLite migrations, FAISS index versioning (Section 13) |
| Batched writes | ✅ | Write queue with batch flush (Section 19) |
| Token-based chunking | ✅ | tiktoken-based chunking (Section 18) |
| Cron / scheduling | ✅ | Standard library cron parser with UTC (Section 16) |
| Sub-agent isolation | ✅ | Read-only memory access for sub-agents (Section 17) |

---

### 20.3 Final Codebase Overview

```
CogniCore/
│
├── config.py                        ← Root config — ALL env vars (Section 10.1)
├── logger.py                        ← Root logger — DEBUG/INFO/WARN/ERROR (Section 10.2)
├── redact.py                        ← Root PII redaction (Section 10.3)
│
├── agent/                           ← Agent Framework (Section 9)
│   ├── main.py                      ← Entrypoint / bootstrap
│   ├── api.py                       ← WebSocket streaming API
│   ├── ingestion.py                 ← REST API for data ingestion
│   ├── sessions.py                  ← Multi-user session management
│   ├── stream.py                    ← EventStream + StreamEvent
│   └── errors.py                    ← Retry policies, fallback chains
│
├── memory/                          ← Memory Layer (Section 2 + 4)
│   ├── types/
│   │   ├── abm.py                   ← Autobiographical Memory (identity)
│   │   ├── wm.py                    ← Working Memory (Redis + JSONL)
│   │   ├── sfm.py                   ← Short-Form Memory (facts, SQLite hot.db)
│   │   ├── lfm.py                   ← Long-Form Memory (docs, SQLite cold.db + FAISS)
│   │   ├── am.py                    ← Associative Memory (Kuzu graph)
│   │   ├── mm.py                    ← Motor Memory (procedures, YAML + Python)
│   │   ├── em.py                    ← Emotional Memory (sentiment, SQLite hot.db)
│   │   ├── pm.py                    ← Prospective Memory (scheduler, SQLite hot.db)
│   │   └── meta.py                  ← Meta Memory (confidence tracking, SQLite hot.db)
│   ├── management/
│   │   ├── mml.py                   ← Memory Management Layer (orchestrator)
│   │   ├── recall.py                ← Hybrid recall (FAISS + BM25 + graph)
│   │   ├── budget.py                ← LLM budget manager
│   │   ├── learning/                ← 10 learning types
│   │   └── maintenance/             ← 7 maintenance daemons
│   └── procedures/                  ← Stored procedures (billing/, collections/)
│
├── analytics/                       ← Analytics Layer (Section 3)
│   └── analytics.py                 ← DuckDB metrics + eval_scores
│
├── response/                        ← Response Layer (Section 5)
│   ├── gate.py                      ← Gate (trivial query filter)
│   ├── pipeline.py                  ← Deep Pipeline orchestrator
│   ├── thinking.py                  ← Query decomposition
│   ├── recall_bridge.py             ← Memory → pipeline bridge
│   ├── decision.py                  ← Decision engine
│   ├── creativity.py                ← Novel insights
│   ├── prediction.py                ← Forecasting
│   ├── synthesis.py                 ← Response composition
│   ├── sub_agents.py                ← Sub-agent spawner
│   └── skills/                      ← Skills execution engine
│
├── control/                         ← Control Systems (Section 6)
│   ├── salience.py                  ← Salience Network (priority scoring)
│   ├── cen.py                       ← Central Executive Network (event loop)
│   ├── dmn.py                       ← Default Mode Network (idle processing)
│   └── governor.py                  ← Governor (guardrails, rate limits, circuit breakers)
│
├── connectors/                      ← Connectors Layer (Section 7)
│   ├── base.py                      ← BaseConnector interface
│   ├── registry.py                  ← Connector registry + lifecycle
│   ├── data/                        ← Data connectors (Snowflake, Databricks, Azure SQL, REST, File)
│   ├── ai/                          ← AI connectors (LLM, Embedder, NLI, Reranker)
│   └── sandbox/                     ← Execution sandboxes (Python, Shell)
│
├── audit/                           ← Audit Layer (Section 8)
│   └── audit.py                     ← Universal logger, @audited decorator
│
├── evals/                           ← Evaluation System (Section 12)
│   ├── runner.py                    ← Eval orchestrator
│   ├── metrics.py                   ← Metric definitions
│   ├── datasets.py                  ← Ground truth datasets
│   ├── reporters.py                 ← Report generation
│   └── types/                       ← Eval implementations (fact_accuracy, recall_quality, etc.)
│
├── migrations/                      ← Versioning & Migration (Section 13)
│   ├── sqlite_migrations.py         ← SQLite schema migrations
│   └── faiss_migrations.py          ← FAISS index versioning
│
├── tests/                           ← Testing (Section 11)
│   ├── unit/                        ← Unit tests (mocked dependencies)
│   ├── integration/                 ← Integration tests (real DBs)
│   ├── e2e/                         ← End-to-end tests
│   ├── fixtures/                    ← Test data
│   ├── mocks.py                     ← Mock connectors
│   └── conftest.py                  ← Pytest fixtures
│
├── prompts/                         ← Prompt Management (Section 10.4)
│   ├── system/                      ← System prompts (identity, thinking, synthesis, decision, etc.)
│   ├── extraction/                  ← Fact/entity/summary extraction prompts
│   ├── skills/                      ← Code gen, SQL gen, analysis prompts
│   └── meta/                        ← Self-reflection, conflict resolution, learning prompts
│
└── README.md
```

**`/datadrive/` (runtime data, all persisted here)**:

```
/datadrive/
├── sqlite/         hot.db, cold.db (all operational tables in one DB each)
├── duckdb/         analytics.duckdb (metrics, eval_results, numbers only)
├── redis/          dump.rdb
├── kuzu/           graph/
├── faiss/          meta.index, sfm.index, lfm.index, am.index, mm.index + *.idmap.json + *.meta.json
├── jsonl/          {uid}/{convo_id}/messages.jsonl + summary.json
├── md/             *.md (LFM knowledge docs)
├── yaml/           {domain}/*.py + *.yaml, _system_generated/
├── models/         embedding model, NLI model, reranker model
├── sandbox/tmp/    temp scripts (auto-cleaned)
├── audit/          {date}.jsonl (append-only, never deleted)
└── logs/           agent.log + rotated backups (agent.log.1, .2, etc.)
```

---

## 21. Deployment & Infrastructure

Single-machine deployment with all components co-located. Zero network latency between components — everything runs on the same box with direct disk access.

---

### 21.1 Architecture Philosophy

**Co-located, Not Distributed**:
- All databases (SQLite, DuckDB, Redis, Kuzu, FAISS) run on the same machine
- No network hops between agent and storage — direct file/socket access
- Raw data files stored on local disk at `/datadrive/`
- Backups handled externally (rsync, cloud sync, or scheduled copies)

**Why Single Machine**:
| Benefit | Explanation |
|---------|-------------|
| **Zero latency** | No network round-trips for DB queries |
| **Simplicity** | One machine to manage, monitor, backup |
| **Consistency** | No distributed transaction complexity |
| **Cost** | One powerful VM cheaper than multiple small ones |
| **Debugging** | All logs, data, processes in one place |

---

### 21.2 Recommended Setup

**Single VM with systemd services** (simplest) or **Docker Compose** (containerized).

#### Option A: Native systemd (Recommended)

```bash
# /etc/systemd/system/cognicore.service
[Unit]
Description=CogniCore Agent
After=network.target redis.service

[Service]
Type=simple
User=cognicore
WorkingDirectory=/opt/cognicore
Environment="PATH=/opt/cognicore/venv/bin"
ExecStart=/opt/cognicore/venv/bin/python -m agent.main
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
# /etc/systemd/system/redis.service (if not using system Redis)
[Unit]
Description=Redis for CogniCore WM

[Service]
Type=simple
ExecStart=/usr/bin/redis-server /etc/redis/redis.conf
Restart=always

[Install]
WantedBy=multi-user.target
```

**Directory Structure**:
```
/opt/cognicore/              ← Application code
├── venv/                    ← Python virtual environment
├── config.py                ← Configuration
├── agent/                   ← Agent code
├── memory/                  ← Memory layer
└── ...

/datadrive/                  ← All persistent data (separate disk recommended)
├── sqlite/                  ← hot.db, cold.db
├── duckdb/                  ← analytics.duckdb
├── redis/                   ← dump.rdb, appendonly.aof
├── kuzu/                    ← graph/
├── faiss/                   ← *.index, *.idmap.json, *.meta.json
├── jsonl/                   ← WM conversations
├── md/                      ← LFM documents
├── yaml/                    ← MM procedures
├── models/                  ← Embedding, NLI, reranker models
├── audit/                   ← Audit logs
└── logs/                    ← Application logs
```

#### Option B: Docker Compose (Containerized)

```yaml
# docker-compose.yml
version: '3.8'

services:
  cognicore:
    build: .
    network_mode: host        # No network overhead
    volumes:
      - /datadrive:/datadrive
      - ./config.py:/app/config.py:ro
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
      - AZURE_OPENAI_KEY=${AZURE_OPENAI_KEY}
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    network_mode: host        # Redis on localhost:6379
    volumes:
      - /datadrive/redis:/data
    command: redis-server --appendonly yes --dir /data
    restart: unless-stopped
```

**Dockerfile**:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

CMD ["python", "-m", "agent.main"]
```

---

### 21.3 Nginx (Reverse Proxy)

Nginx runs on the same machine, proxying to localhost ports.

```nginx
# /etc/nginx/sites-available/cognicore
server {
    listen 443 ssl http2;
    server_name agent.example.com;

    ssl_certificate /etc/letsencrypt/live/agent.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/agent.example.com/privkey.pem;

    # WebSocket
    location /ws {
        proxy_pass http://127.0.0.1:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 3600s;
    }

    # REST API
    location /api/ {
        proxy_pass http://127.0.0.1:8080/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /health {
        proxy_pass http://127.0.0.1:8080/ingest/health;
    }
}

server {
    listen 80;
    server_name agent.example.com;
    return 301 https://$server_name$request_uri;
}
```

---

### 21.4 Hardware Requirements

**Recommended Specs**:
- **CPU**: 8+ cores (embedding, LLM calls are CPU-bound when waiting)
- **RAM**: 32GB+ (FAISS indexes load into memory, Redis in-memory)
- **Storage**: 500GB+ NVMe SSD (fast random I/O for SQLite/DuckDB)
- **GPU**: Optional — only if running local LLM or large embedding models

**Storage Sizing**:

| Component | Growth Rate | Retention | Est. 1 Year |
|-----------|-------------|-----------|-------------|
| SQLite (hot.db) | ~10MB/day | Forever | ~4GB |
| SQLite (cold.db) | ~50MB/day | Forever | ~20GB |
| DuckDB (analytics) | ~5MB/day | Forever | ~2GB |
| FAISS indexes | ~1MB/1000 vectors | Forever | ~10GB |
| Audit logs | ~20MB/day | 1 year | ~8GB |
| WM JSONL | ~5MB/day | 30 days | ~150MB |
| Models | Static | Forever | ~2GB |
| **Total** | | | **~50GB** |

**Disk Layout** (recommended):
```
/              ← OS + application (100GB SSD)
/datadrive     ← All data (separate 500GB+ NVMe, or mounted volume)
```

---

### 21.5 Performance Benefits

Since everything is co-located:

| Operation | Latency | Why |
|-----------|---------|-----|
| SQLite query | <1ms | Direct file access, no network |
| Redis get/set | <0.1ms | Unix socket or localhost |
| FAISS search | <10ms | Memory-mapped, no serialization |
| DuckDB analytics | <50ms | Direct file, columnar scans |
| Kuzu graph query | <5ms | Embedded database |

**No need for**:
- Connection pooling (direct access)
- Network timeouts (localhost)
- Distributed locks (single process)
- Service discovery (everything on localhost)

---

### 21.6 Monitoring & Alerting

**Metrics to Monitor**:

| Metric | Source | Alert Threshold |
|--------|--------|-----------------|
| Response latency (p95) | Analytics | >5s |
| LLM call latency (p95) | Analytics | >10s |
| Memory recall latency | Analytics | >500ms |
| Error rate | Audit | >1% |
| Redis fallback mode | WM health | Any |
| Disk usage | System | >80% |
| Memory usage | System | >85% |
| FAISS index size | Analytics | >1M vectors |

**Recommended Stack**:
- **Metrics**: Prometheus + Grafana
- **Logs**: Loki or ELK (ingest audit JSONL)
- **Alerts**: Alertmanager or PagerDuty

**Prometheus Exporter** (add to agent):

```python
# agent/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

REQUEST_COUNT = Counter('cognicore_requests_total', 'Total requests', ['type'])
REQUEST_LATENCY = Histogram('cognicore_request_latency_seconds', 'Request latency', ['type'])
MEMORY_SIZE = Gauge('cognicore_memory_size', 'Memory size', ['type'])
FAISS_VECTORS = Gauge('cognicore_faiss_vectors', 'FAISS vector count', ['index'])

def start_metrics_server(port: int = 9090):
    start_http_server(port)
```

---

### 21.7 Backup Strategy

All data lives in `/datadrive/` — backup this directory externally.

**Backup Options**:

| Method | Best For | Command |
|--------|----------|---------|
| **rsync** | Local/remote server | `rsync -avz /datadrive/ backup-server:/backups/cognicore/` |
| **Cloud sync** | S3/GCS/Azure Blob | `rclone sync /datadrive/ remote:cognicore-backup/` |
| **Snapshot** | VM-level backup | Cloud provider snapshot (AWS EBS, Azure Disk) |
| **tar + upload** | Manual/scheduled | See script below |

**What Gets Backed Up**:

```
/datadrive/
├── sqlite/      ← hot.db, cold.db (critical — all operational data)
├── duckdb/      ← analytics.duckdb (metrics, evals)
├── redis/       ← dump.rdb (WM state, can regenerate)
├── kuzu/        ← graph/ (associations)
├── faiss/       ← *.index + *.idmap.json + *.meta.json (can rebuild from SQLite)
├── jsonl/       ← WM conversations (can regenerate from Redis)
├── md/          ← LFM source documents (important)
├── yaml/        ← MM procedures (important)
├── models/      ← Downloaded models (can re-download)
├── audit/       ← Compliance logs (important, append-only)
└── logs/        ← App logs (optional)
```

**Priority for Recovery**:
1. `sqlite/` — All facts, documents, schedules, pointers
2. `md/` + `yaml/` — Source documents and procedures
3. `audit/` — Compliance trail
4. Everything else can be regenerated or re-downloaded

**Backup Script**:

```bash
#!/bin/bash
# backup.sh - Run via cron: 0 * * * * /opt/cognicore/backup.sh

set -e
TIMESTAMP=$(date +%Y-%m-%d_%H%M)
BACKUP_DIR="/backups/$TIMESTAMP"

mkdir -p "$BACKUP_DIR"

# SQLite online backup (safe while running)
sqlite3 /datadrive/sqlite/hot.db ".backup '$BACKUP_DIR/hot.db'"
sqlite3 /datadrive/sqlite/cold.db ".backup '$BACKUP_DIR/cold.db'"

# Copy other critical files
cp /datadrive/duckdb/analytics.duckdb "$BACKUP_DIR/"
cp -r /datadrive/faiss "$BACKUP_DIR/"
cp -r /datadrive/kuzu "$BACKUP_DIR/"
cp -r /datadrive/md "$BACKUP_DIR/"
cp -r /datadrive/yaml "$BACKUP_DIR/"
cp -r /datadrive/audit "$BACKUP_DIR/"

# Compress
tar -czf "/backups/cognicore-$TIMESTAMP.tar.gz" -C "$BACKUP_DIR" .
rm -rf "$BACKUP_DIR"

# Upload to remote (choose one)
# rsync -avz "/backups/cognicore-$TIMESTAMP.tar.gz" backup-server:/backups/
# aws s3 cp "/backups/cognicore-$TIMESTAMP.tar.gz" s3://cognicore-backups/
# rclone copy "/backups/cognicore-$TIMESTAMP.tar.gz" remote:cognicore-backup/

# Cleanup old local backups (keep 7 days)
find /backups -name "cognicore-*.tar.gz" -mtime +7 -delete
```

---

### 21.8 Disaster Recovery

**Recovery is simple** — restore `/datadrive/` and start the agent.

**Steps**:

1. **Provision new machine** with same specs
2. **Restore data**:
   ```bash
   # From rsync backup
   rsync -avz backup-server:/backups/cognicore/ /datadrive/
   
   # Or from tar archive
   tar -xzf cognicore-latest.tar.gz -C /datadrive/
   ```
3. **Install and start**:
   ```bash
   cd /opt/cognicore
   pip install -r requirements.txt
   systemctl start cognicore redis nginx
   ```
4. **Verify**:
   ```bash
   curl http://localhost:8080/health
   ```

**RTO/RPO**:
- **RPO** (data loss): Depends on backup frequency (hourly = 1 hour max loss)
- **RTO** (downtime): ~15 minutes (restore + start services)

---

### 21.9 Security Checklist

| Item | Implementation |
|------|----------------|
| TLS termination | Nginx with Let's Encrypt |
| API authentication | OAuth2 via Nginx (external) |
| Secrets management | Environment variables (not in code) |
| Network isolation | Private subnet for Redis, DBs |
| PII protection | `redact.py` on all writes |
| Audit trail | Append-only JSONL logs |
| Code execution | Sandboxed with AST validation |
| Rate limiting | Governor + Nginx |
| Input validation | Schema validation on all APIs |
| Dependency scanning | Dependabot / Snyk in CI |
