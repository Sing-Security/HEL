# HEL Crate — Parser and Evaluator

Status: OPEN (Apache-2.0)  
SPDX-License-Identifier: Apache-2.0

Purpose
- HEL (Hermes Expression Language) is an auditable, deterministic expression language used to encode security rules and domain packages.
- This crate provides the open reference implementation used by the Hermes rule engine: lexer/parser (pest-based), AST definitions, type/schema helpers, built-in registry interfaces, and a deterministic evaluator implementation suitable for embedding in higher-level services.

Scope and boundaries
- In-scope (open): parsing, AST, type/schema helpers, evaluation primitives, the open builtins registry interface, and package/schema loaders for HEL domain packages.
- Out-of-scope (closed / product): product business rules, proprietary scoring algorithms, vendor fingerprint heuristics, and any product-specific built-ins. Those belong in product crates or closed built-ins loaded via the builtins provider interface.

High-level architecture
- Parsing: pest-based grammar (hel.pest). The crate exposes a parser type for unit testing and integration.
- AST & types: typed AST nodes (booleans, numbers/floats, identifiers, attribute access, comparisons, boolean combinators, lists/maps, function calls). Schema support and package manifests live under `schema/`.
- Evaluation: deterministic, pure evaluation against an immutable context with trace support exported from `trace`.
- Builtins: pluggable via the `BuiltinsProvider` / `BuiltinsRegistry` interfaces so closed/proprietary functions can be injected at runtime.

Determinism & safety
- The public evaluator is designed to be pure (no global mutable state or I/O during evaluation).
- `unsafe` usage is to be minimized, justified, and wrapped in safe APIs. Libraries in the ForgeCore platform should avoid exposing `unsafe` to consumers.
- The crate does not currently depend on arbitrary regex engines; avoid regex constructs that could cause super-linear behavior. Any pattern-matching builtins must be implemented with bounded execution characteristics.

Error handling
- Parse errors surface structured diagnostics with span information where possible.
- Type and schema errors are reported as explicit error types; evaluation returns `Result<..., Error>`; libraries must avoid panics in public APIs.

API surface (snapshot)
- Modules exported by the crate:
  - `hel::schema` — Package manifest, SchemaPackage loader, type definitions and helpers.
    - key types: `PackageManifest`, `SchemaPackage`, `Schema`, `TypeEnvironment` (see `schema/package.rs`).
  - `hel::builtins` — Builtin provider interfaces.
    - key traits/types: `BuiltinsProvider`, `BuiltinsRegistry`, `CoreBuiltinsProvider`, `BuiltinFn`.
  - `hel::trace` — Evaluation tracing helpers.
    - key items: `EvalTrace`, `evaluate_with_trace`, `AtomTrace`.
  - Public data fact types (examples found in `src/lib.rs`):
    - `BinaryInfo`, `SecurityFlags`, `SectionInfo`, `ImportInfo`, `TaintFlow`, `FunctionCall`, `MemoryOperation`, `SymQueryRequest`.
  - Parser type (pest): `HelParser` (generated via `pest_derive`).
- Note: the crate exposes types and interfaces rather than a single high-level `Compiler`/`Evaluator` type in the current code. Do not rely on symbols that are not present in `src/lib.rs`.

Documentation fixes and removals
- Remove or replace the code snippet that references `Compiler::default().compile(...)` and `Evaluator::default().eval(...)` with a conceptual usage paragraph. The code snippet in the existing README refers to types that are not exported from the current crate surface (no `Compiler`/`Evaluator` symbols were found).
- Remove the assertive statement that "Regex engine uses RE2-like semantics" — the crate currently does not depend on a regex engine. Replace with guidance: "Avoid unbounded regexes; any pattern-match builtins must ensure bounded, deterministic evaluation."
- Ensure the README does not include any product names, product-specific rule examples, or policy logic. Platform crates must stay product-agnostic.

Examples and how-to (conceptual)
- The crate provides parsing and evaluation primitives and a trace-based evaluator for integration. Typical integration steps:
  1. Load HEL expression text (source).
  2. Use the parser to produce an AST (parser is pest-based).
  3. Create an immutable evaluation context (domain facts are represented via the schema types; packages can be loaded via `SchemaPackage::from_directory`).
  4. Evaluate the AST with the builtins registry (open builtins) or a `BuiltinsProvider` implementation for closed builtins. The crate exports `evaluate_with_trace` to obtain both the evaluation result and an execution trace suitable for audits.

Extensibility
- `BuiltinsProvider` is the supported extension point for closed or product-specific functions (loaded at runtime by the embedding product).
- Domain packages: `hel-package.toml` + schema files are supported; packages are loaded via the schema package loader and merged into a `Schema` object.

Licensing
- This crate and its open built-ins are Apache-2.0. Closed builtins and product rule packs must be delivered separately and must not be included in the open crate.

Contribution guidelines
- Follow GEMINI rules: determinism, auditability, minimal `unsafe`, explicit error types, and strict separation of platform vs. product logic.
- When adding built-ins to the open crate: ensure they are generic, deterministic, and useful across products. Vendor- or product-specific built-ins must be implemented in closed crates and injected via `BuiltinsProvider`.

Further reading
- See the local documentation (`docs/USAGE.md`, `docs/SCHEMA.md`) and the crate's `src/schema` module for package/schema examples and manifest format.
