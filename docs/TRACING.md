# HEL — Evaluation Tracing (Audit Guide)

Status: OPEN (Apache-2.0)  
SPDX-License-Identifier: Apache-2.0

Purpose
- Define the stable, auditable trace shape produced by the HEL evaluator.
- Give implementers and integrators clear guidance for emitting, recording, and consuming traces for compliance, debugging, and replay.
- Enforce GEMINI constraints: determinism, explainability, and minimal leakage of sensitive data.

Scope
- This document describes the trace model emitted by `evaluate_with_trace`-style evaluators and the responsibilities of the evaluator, builtin implementations, and host integrators.
- It is NOT an implementation spec for a particular tracing API in code, but a normative description of trace contents, serialization, and audit/evidence practices.

Principles
1. Determinism: traces for the same inputs (expression text, schema, resolver inputs, builtin registry) must be byte-for-byte identical when serialized with the canonical serialization rules below.
2. Explainability: every decision point must be traceable to a span in the expression and to input values.
3. Minimal exposure: traces should avoid including raw secrets or PII unless explicitly allowed by policy; prefer canonical hashes for sensitive blobs.
4. Versioning: traces must include version identifiers for expression text, schema package(s), and builtin registry so results are reproducible.

Overview of trace emission responsibilities
- Evaluator:
  - Orchestrates evaluation and collects entries for all evaluation steps.
  - Emits top-level metadata (expression id/hash, schema ids, registry snapshot).
  - Serializes the final trace using canonical rules.
- Builtin implementations:
  - Must return execution metadata to the evaluator (input summary, output summary, error if any).
  - Must not perform I/O during trace emission. Any additional metadata (model version, implementation id) must be declarative and included in return metadata fields.
- Host integrator:
  - Provides deterministic resolver and builtin registry.
  - Persists trace along with expression source and schema package versions for audit.

Trace model (logical)
A trace is a deterministic, ordered list of entries with top-level metadata. The evaluator MUST produce a single top-level JSON object with the following fields:

- `meta` — object: top-level metadata about the evaluation (see below).
- `entries` — array of trace entry objects in evaluation order (see entry types).
- `result` — final evaluation result summary (repeated for convenience).
- `error` — optional top-level evaluation error (only present if evaluation failed catastrophically).

Top-level `meta` fields (required)
- `expr_id` (string): canonical identifier of the expression text (e.g., sha256 hex).
- `expr_text` (string): expression source text (optional for privacy; if omitted, must still provide `expr_id`).
- `expr_version` (string): if the host versioned expressions, include version tag.
- `schema_snapshots` (array): list of objects `{ name: string, version: string, sha256: string }` for each schema package used.
- `registry_snapshot` (array): list of `{ name: string, version: string }` for builtins available during evaluation.
- `timestamp_utc` (string, RFC3339): optional; only if host pins clock deterministically for reproducibility.
- `evaluator_version` (string): version or git sha of the evaluator implementation.
- `trace_format_version` (string): version of this trace shape (bump when changing fields/semantics).

Entry types (each entry MUST include `type` and `span` where relevant)
- `eval_start` — marks beginning of evaluation.
  - `{ "type": "eval_start", "timestamp_ms": number }`
- `literal_eval` — evaluation of a literal or constant.
  - `{ "type":"literal_eval", "span": {"start":int,"end":int}, "value": <ValueSummary> }`
- `identifier_resolve` — resolver lookup for an attribute (object.field).
  - `{ "type":"identifier_resolve", "span": {...}, "name": "binary.arch", "resolved": <ValueSummary | null>, "method": "resolver" }`
- `operation` — primitive ops (comparisons, arithmetic, boolean ops).
  - `{ "type":"operation", "span": {...}, "op": "&&" | "==" | "+", "left": <ValueSummary>, "right": <ValueSummary>, "result": <ValueSummary> }`
- `builtin_call` — a call into a builtin. This is mandatory for every builtin invocation.
  - `{ "type":"builtin_call", "span": {...}, "builtin": "hash.sha256", "builtin_version": "0.2.1", "inputs": [<ValueSummary>], "output": <ValueSummary|{error:...}>, "duration_ms": number }`
- `match_case` — evaluation of a `match` arm (optional).
- `let_bind` — creation of a let-bound value (optional).
- `eval_end` — marks end of evaluation; includes final value summary.
  - `{ "type":"eval_end", "timestamp_ms": number, "value": <ValueSummary> }`
- `error` — evaluation error not tied to builtin (type mismatch, overflow).
  - `{ "type":"error", "span": {...}?, "code": "type_error" | "overflow" | "...", "message": "..." }`

ValueSummary (canonical compact representation)
- To avoid large payloads, every `value` or `*Summary` field should be one of:
  - Primitive:
    - `{ "kind": "bool", "v": true | false }`
    - `{ "kind": "number", "v": 1.234 }` (IEEE-754 double represented as JSON number)
    - `{ "kind": "string", "v": "..." }` (UTF-8)
  - Null:
    - `{ "kind": "null" }`
  - Bytes (binary blobs):
    - `{ "kind": "bytes", "sha256": "hex...", "len": 123 }` — prefer canonical hash instead of raw bytes; if raw included, must be base64 and flagged in meta.
  - List:
    - `{ "kind": "list", "len": N, "preview": [<ValueSummary>], "sha256": "optional hash-of-canonical-serialization" }` — `preview` is bounded (host-configurable, default 8 items).
  - Map/Object:
    - `{ "kind": "map", "len": N, "preview": [{"k": "key", "v": <ValueSummary>}], "sha256": "optional" }`
  - Error/Err result:
    - `{ "kind": "err", "code": "parse_error", "message": "..." }`

Rationale: ValueSummary gives auditors visibility into intermediate values while keeping traces bounded and avoiding accidental leakage of large binary content. If policy allows raw inclusion, indicate in meta and include base64 under a dedicated field.

Canonical serialization rules (determinism)
- Serialization MUST be JSON with:
  - Object keys sorted lexicographically.
  - No extra whitespace (minified canonical JSON).
  - Floating point numbers serialized with a deterministic formatter (e.g., shortest round-trip decimal that parses back to same binary).
  - All maps inside `ValueSummary` must present `preview` items sorted by key when including more than one.
- For JSON canonicalization, use a stable algorithm (example: RFC 8785 JSON Canonicalization Scheme) or an internal project standard — document which one and include its identifier in `meta.trace_serialization`.
- When hashing values for `sha256`, canonicalize the value using the same canonical JSON rules before hashing.

Trace example (full evaluation)
- Example minimal trace for an expression `entropy(file.bytes) > 7.0` (values abbreviated).

```desmond/forgecore/crates/hel/docs/TRACING.md#L1-120
{
  "meta": {
    "expr_id": "sha256:3a7f...e9",
    "expr_text": "entropy(file.bytes) > 7.0",
    "schema_snapshots": [
      { "name": "com.example.binary.schema", "version": "0.1.0", "sha256": "a1b2..." }
    ],
    "registry_snapshot": [
      { "name": "core", "version": "0.9.0" },
      { "name": "desmond-closed", "version": "1.5.0" }
    ],
    "evaluator_version": "hel-eval@0.5.3",
    "trace_format_version": "1"
  },
  "entries": [
    { "type":"eval_start", "timestamp_ms": 1670000000000 },
    { "type":"identifier_resolve", "span": {"start":8,"end":18}, "name":"file.bytes",
      "resolved": { "kind": "bytes", "sha256": "deadbeef...", "len": 4096 }, "method":"resolver" },
    { "type":"builtin_call", "span": {"start":0,"end":20}, "builtin":"entropy", "builtin_version":"0.1.0",
      "inputs":[ { "kind":"bytes", "sha256":"deadbeef...", "len":4096 } ],
      "output": { "kind":"number", "v": 7.8123 }, "duration_ms": 1.2 },
    { "type":"operation", "span": {"start":0,"end":20}, "op": ">", "left": { "kind":"number","v":7.8123 }, "right": { "kind":"number","v":7.0 }, "result": { "kind":"bool","v": true } },
    { "type":"eval_end", "timestamp_ms": 1670000000001, "value": { "kind":"bool", "v": true } }
  ],
  "result": { "kind":"bool", "v": true }
}
```

Trace entry guidelines for builtin authors
- Include `builtin_version` and a short `implementation` identifier in provider metadata.
- Provide deterministic `inputs` and `output` value summaries.
- If the builtin computes or uses a model, add a metadata map to the trace entry with `{ "model": { "id": "foo", "version": "v1" } }`.
- Avoid returning raw large blobs; prefer `sha256` and length. If raw inclusion is required, the evaluator should be configured to redact or encrypt trace fields.

Privacy, PII, and secrets
- Traces SHOULD NOT contain raw secrets or PII by default.
- If the host policy allows inclusion:
  - Record an explicit consent flag in `meta` (e.g., `meta.include_sensitive: true`) and ensure secure storage/encryption of the persisted trace.
  - Use redaction markers for sensitive fields, e.g., `{ "kind":"string", "redacted": true, "reason": "PII" }`.
- For binary blobs (certificates, binaries), prefer canonical hashes plus length; provide a path to an encrypted artifact store rather than embedding raw bytes.

Storage, retention, and evidence
- Persist, per evaluation:
  - The trace JSON (canonicalized).
  - The expression source text (or ensure `expr_id` can be expanded).
  - Schema package manifests or their canonical hashes.
  - Registry snapshot (builtin names & versions).
  - Resolver input snapshot (host facts) — serialized under the same canonical rules.
- Evidence retention: follow organizational compliance policy. For regulated environments, provide tamper-evidence (signed manifest + trace) and store checksums in an append-only log.
- Consider separating trace storage tiers:
  - Short-term: full traces for debugging (retained temporarily).
  - Long-term: compact evidence records (hashes, metadata) for audit replay.

Trace consumption (replay)
- To replay an evaluation deterministically:
  1. Retrieve `expr_text` and/or verify `expr_id` against stored text.
  2. Reconstruct resolver inputs from persisted snapshot.
  3. Re-create the builtin registry state (by using the recorded `registry_snapshot` to load exact provider versions).
  4. Run the evaluator with the same pinned evaluator version where possible, and compare the produced canonical trace to stored trace (byte-for-byte).
- Store enough metadata to map provider versions to published artifacts or closed-provider packages.

Performance considerations
- Traces can be heavy; allow host-configurable sampling and level-of-detail:
  - `trace_level: minimal | standard | verbose`
  - `preview_size`: number of items to include in list/map previews (default 8).
- For high-throughput paths, prefer `standard` with previews and hash-only for large blobs.
- Instrument trace emission to measure overhead; include `duration_ms` on builtin entries and evaluator-level timing so hosts can account for tracing cost.

Implementation notes for evaluators
- Build trace entries as you evaluate in left-to-right order; do not reorder entries after the fact.
- Keep a deterministic buffer/collector and only write the final canonical JSON at the end (or stream deterministic fragments if required, but ensure canonical final record).
- Provide configuration options:
  - `include_raw_bytes: bool` (default false)
  - `trace_level: enum`
  - `preview_length: usize`
  - `max_entry_count` and `max_trace_size_bytes` to avoid unbounded traces
- When hitting limits, include a special entry:
  - `{ "type":"trace_truncated", "reason":"size_limit", "max_bytes": 1048576 }`

Example: consumer assertions in tests
- Tests should assert:
  - `meta.expr_id` matches expected hash for expr text.
  - `registry_snapshot` contains expected builtin names and versions.
  - For a rule that should be true, final `result.kind == "bool" && result.v == true`.
  - Presence of specific `builtin_call` entries and expected input/output summaries.

JSON Schema (recommended minimal shape)
- Provide and maintain a machine-readable JSON schema for trace validation. Example conceptual shape (not normative here):
  - `meta` object as specified above.
  - `entries` array of objects each with `type` and typed fields for each supported entry type.
- Implement CI validation that ensures traces conform to the JSON schema whenever trace-emitting code changes.

Appendix — Quick checklist for host integrators
- [ ] Ensure evaluator and builtin providers are versioned and recordable.
- [ ] Configure trace level appropriate for environment (dev/prod).
- [ ] Persist expression text or ensure stable mapping from `expr_id` to text.
- [ ] Persist schema package manifests and builtin registry snapshot.
- [ ] Enforce privacy policy: do not include PII by default.
- [ ] Include canonicalization algorithm identifier in `meta.trace_serialization`.
- [ ] Add tests that assert trace shape and presence of key builtin call entries.
