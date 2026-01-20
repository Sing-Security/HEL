# HEL — Introduction

This document is the developer-facing introduction to the HEL (Hermes Expression Language) crate. It is written for engineers who will embed HEL in host products, extend the language with built-ins, author or maintain schema packages, or contribute to the crate itself.

I designed this introduction to orient you quickly and point you to the other detailed documents in this directory.

## Who this is for
- You are embedding HEL into a product (Hermes rule engine, Stratus workflows, or a custom host).
- You are implementing built-ins (open or closed) and need to understand extension points and audit boundaries.
- You are authoring schema packages so HEL expressions can be evaluated against typed domain facts.
- You are maintaining or testing the `hel` crate itself.

## Purpose of this docs set
- Provide clear, audit-ready developer documentation so hosts can safely embed and extend HEL.
- Preserve GEMINI constraints: determinism, auditability, minimal `unsafe`, explicit errors, and strict separation of platform vs product logic.
- Keep the crate product-agnostic: do not place product-specific business rules or proprietary scoring logic here.

## Files in this directory
- `INTRO.md` — This file (high-level orientation and reading order).
- `USAGE.md` — Quickstart: parsing, AST, evaluation, and common embedding patterns.
- `SCHEMA.md` — Schema package format, loaders, and examples.
- `BUILTINS.md` — Builtins provider/registry interfaces and guidance for open vs closed builtins.
- `TRACING.md` — Evaluation tracing API, trace formats, and audit guidance.
- `CONTRIBUTING.md` — Development, testing, and contribution guidelines aligned with GEMINI.
- `REFERENCE.md` — Short reference and entry points to the language spec and related docs.

In addition, consult:
- The language specification at the repository top-level: `docs/hel.md`
- The `binx` integration notes: `docs/binx/HEL.md`
- The crate README: `README.md`

## Quick conceptual overview
HEL is a deterministic, side-effect-free expression language meant for writing auditable heuristics and transforms over structured security data.

Key technical boundaries you must respect when embedding or extending HEL:
- Deterministic evaluation: no I/O, no randomness, no hidden global state in evaluation paths.
- Open vs closed built-ins: the crate exposes a `BuiltinsProvider`/`BuiltinsRegistry` interface so host products can inject proprietary functions at runtime. Keep product logic out of the open crate.
- Schema-driven evaluation: domain types (packages) may be loaded and merged into a `Schema` used by resolvers to map host facts into HEL `Value` instances.
- Tracing & audit: engines must support a trace export that ties every evaluation result back to expression spans, intermediate values, and builtin calls.

Typical integration steps (conceptual)
1. Load a HEL expression source (string).
2. Parse to produce an AST using the crate's parser.
3. Prepare an immutable evaluation context (resolver + schema, plus a builtin registry/provider).
4. Evaluate the AST using the evaluator API that emits both the result and an evaluation trace suitable for audits.
5. Record result, trace, and the versioned expression source for reproducibility.

Note: prefer the documented APIs (`HelParser`, `evaluate_with_trace`, `BuiltinsProvider`, schema loaders) rather than any undocumented helper types. The public evaluator is intentionally pure; any builtin that requires external access must be implemented by the host and injected.

## Recommended reading order
1. `USAGE.md` — get a hands-on sense for parsing and evaluating expressions in host code.
2. `SCHEMA.md` — learn how to author and load schema packages (type-checking guidance and package manifest format).
3. `BUILTINS.md` — see the extension points and how to separate open builtins from closed/product builtins.
4. `TRACING.md` — understand the trace format and how to export audit evidence.
5. `CONTRIBUTING.md` — development guidelines, tests, and the GEMINI-aligned rules you must follow.
6. `REFERENCE.md` and `docs/hel.md` — language reference and normative spec.

## Quick embedding checklist
Before shipping an integration, verify the following:
- Determinism: All inputs are explicit and the builtin registry is deterministic for identical inputs.
- Auditability: Engine emits traces for every evaluation and the expression source is versioned and stored alongside results.
- No product logic leaks: Closed builtins are implemented outside this crate and injected via `BuiltinsProvider`.
- Error handling: The host treats parse/type/eval errors as structured data (no panics) and records them with spans.
- Tests & fixtures: Rules are covered with deterministic unit tests and fixtures; floating point comparisons and time-related functions are exercised explicitly.

## Auditing & evidence
When collecting audit evidence for a rule evaluation you should record:
- The exact expression text and its canonical identifier (hash + version).
- The `Schema` version(s) and package manifests in effect during evaluation.
- The builtin registry version (or a list of builtin names + versions) used during that evaluation.
- The evaluation trace emitted by the engine (see `TRACING.md`).
- Host inputs (facts) used by the resolver, serialized with stable deterministic ordering.

This information is required so evaluations can be reproduced and validated in compliance contexts.

## Contribution & extension principles (summary)
- Follow GEMINI: determinism, explainability, auditability, minimal `unsafe`.
- Open crate only contains generic, broadly useful builtins. Product-specific builtins must be closed and injected at runtime.
- All public APIs must return `Result<T, E>` on failure; no panics in library code.
- Add tests for any language or evaluation changes and include trace-based assertions where appropriate.
- Document any non-obvious deterministic behavior (floating-point semantics, regex limits, list/map ordering) in `REFERENCE.md` and the top-level spec at `docs/hel.md`.

## Contact points and follow-ups
- If you plan to author schema packages, start with `SCHEMA.md` and the examples referenced there.
- If you plan to write closed builtins, read `BUILTINS.md` for provider API and packaging guidance.
- For trace consumers and auditors, read `TRACING.md` to understand the trace shapes the evaluator emits.

---

Next step: open `USAGE.md` to get an immediate, conceptual quickstart and code patterns for parsing, schema loading, resolver wiring, builtin injection, and trace collection.