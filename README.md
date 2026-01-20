# HEL — Heuristics Expression Language

Status: OPEN — Apache-2.0  
SPDX-License-Identifier: Apache-2.0

Overview
- HEL (Internally Hermes Expression Language) is a small, deterministic, auditable expression language and reference implementation.
- This crate implements the open core: a pest-based parser, a compact typed AST, deterministic evaluator(s), a pluggable builtins registry, schema/package loaders for domain types, and a trace facility that produces stable, auditable evaluation traces.
- The crate is intentionally product-agnostic: domain-specific or proprietary built-ins and rule packs should be implemented and shipped separately and injected at runtime via the builtins provider interface.

Goals
- Determinism: evaluation order and iteration are stable (stable maps, deterministic traces).
- Auditability: fine-grained atom-level traces that show resolved inputs and atom results.
- Extensibility: runtime injection of domain built-ins via a clear provider/registry API.
- Minimal surface area: provide primitives (parser, AST, evaluator, trace, schema loader) rather than a monolithic runtime.

What this crate provides (public capabilities)
- Parsing
  - A pest grammar (`hel.pest`) and parser type `HelParser`.
  - Top-level parse helper: `parse_rule(condition: &str) -> AstNode`.
- AST
  - `AstNode` variants: `Bool`, `String`, `Number`, `Float`, `Identifier`, `Attribute`, `Comparison`, `And`, `Or`, `ListLiteral`, `MapLiteral`, `FunctionCall`.
  - Comparators supported: `==`, `!=`, `>`, `>=`, `<`, `<=`, `CONTAINS`, `IN`.
- Evaluation
  - Resolver-based evaluation: `evaluate_with_resolver(condition: &str, resolver: &dyn HelResolver) -> Result<bool, EvalError>`.
  - Evaluation with builtins: `evaluate_with_context(condition: &str, resolver: &dyn HelResolver, builtins: &builtins::BuiltinsRegistry)`.
  - Runtime `Value` model: `Null`, `Bool`, `String(Arc<str>)`, `Number(f64)`, `List(Vec<Value>)`, `Map(BTreeMap<Arc<str>, Value>)`.
  - `HelResolver` trait for embedding hosts to supply attribute values (object.field).
- Builtins and extensibility
  - `BuiltinsProvider` trait and `BuiltinsRegistry` for namespace-aware function dispatch.
  - `BuiltinFn` type: pure, deterministic functions that map argument `Value`s to a `Result<Value, EvalError>`.
  - `CoreBuiltinsProvider` included with a small set of generic functions (`core.len`, `core.contains`, `core.upper`, `core.lower`).
- Trace & audit
  - `evaluate_with_trace(condition, resolver, Option<&BuiltinsRegistry>) -> Result<EvalTrace, EvalError>`.
  - `EvalTrace` contains a deterministic list of `AtomTrace` entries and a sorted list of `facts_used()`.
  - Pretty-print helpers to produce deterministic, human-readable traces.
- Schema and package system
  - Schema parser and in-memory `Schema` representation (`FieldType`, `TypeDef`, `FieldDef`).
  - Package manifest type `PackageManifest` (`hel-package.toml`), `SchemaPackage`, and `PackageRegistry` for loading and resolving packages into a `TypeEnvironment`.
  - Deterministic package resolution and type merging with clear collision detection.

Quick usage examples
- Parse an expression into an AST:
```/dev/null/example_parse.rs#L1-20
use hel::parse_rule;

let ast = parse_rule("binary.format == \"elf\" AND security.nx_enabled == true");
// `ast` is an `AstNode` representing the parsed expression
```

- Evaluate with a simple resolver:
```/dev/null/example_eval.rs#L1-40
use hel::{evaluate_with_resolver, HelResolver, Value};

struct MyResolver;
impl HelResolver for MyResolver {
    fn resolve_attr(&self, object: &str, field: &str) -> Option<Value> {
        match (object, field) {
            ("binary", "format") => Some(Value::String("elf".into())),
            ("security", "nx_enabled") => Some(Value::Bool(true)),
            _ => None,
        }
    }
}

let resolver = MyResolver;
let result = evaluate_with_resolver(r#"binary.format == "elf""#, &resolver)?;
assert!(result);
```

- Evaluate with builtins and capture a trace:
```/dev/null/example_trace.rs#L1-60
use hel::{evaluate_with_trace, HelResolver, builtins::BuiltinsRegistry, builtins::CoreBuiltinsProvider};

let mut registry = BuiltinsRegistry::new();
registry.register(&CoreBuiltinsProvider)?;

struct MyResolver;
impl HelResolver for MyResolver {
    fn resolve_attr(&self, object: &str, field: &str) -> Option<hel::Value> { /* ... */ unimplemented!() }
}

let trace = evaluate_with_trace("core.len([1,2,3]) == 3", &MyResolver, Some(&registry))?;
println!("{}", trace.pretty_print()); // deterministic, human-friendly audit trail
```

Design notes and important details
- Determinism
  - Internal maps use `BTreeMap` and lists are iterated stably to ensure deterministic behavior across runs.
  - Traces and `facts_used()` are sorted to make audit logs stable.
- Pure builtins
  - Builtins must be pure and deterministic; they must not perform unbounded I/O or rely on global mutable state. The registry enforces namespace isolation and stable ordering.
- Error handling
  - Public evaluation functions return `Result<..., EvalError>`. `EvalError` covers parse errors, type mismatches, unknown attributes, and invalid operations.
- Limits & omissions
  - The core language focuses on declarative expressions and comparisons. It does not provide arithmetic operators (`+`, `-`, `*`, `/`) beyond numeric comparisons in the current implementation.
  - Function calls require a `BuiltinsRegistry` in the evaluation context. Without it, invoking `FunctionCall` yields an `InvalidOperation` error.
  - The crate exposes primitives (parser, AST, evaluator, trace, schema loader) and intentionally does not provide a single monolithic "compiler" or product-specific rule engine.
- Performance & safety
  - The evaluator uses `f64` for runtime numbers; integer literal parsing persists `u64` in the AST then converts as needed to `Value::Number(f64)`.
  - Avoid unbounded regexes in any custom builtins. The crate itself does not depend on a regex engine; pattern-match builtins must ensure bounded, deterministic execution.

Documentation and where to look next
- Read the `src` modules to get API-level details:
  - `hel::schema` — package manifest, `SchemaPackage`, schema parsing helpers.
  - `hel::builtins` — provider/registry API and `CoreBuiltinsProvider`.
  - `hel::trace` — trace capture and pretty-print helpers.
  - `hel::parse_rule` and the AST in `src/lib.rs`.
- Local docs: `docs/USAGE.md` and `docs/SCHEMA.md` (examples and schema/package format).
- Tests in `src/*` demonstrate intended semantics and edge-case behavior (NaN handling, builtins, trace order, package registry collision detection).

Contributing
- Follow these principles when contributing:
  - Preserve determinism and auditability.
  - Keep open built-ins generic and product-agnostic.
  - When adding features that affect evaluation semantics, add deterministic tests and trace-based examples.
  - Avoid exposing `unsafe` in public APIs unless strictly necessary and justified with clear documentation.

License
- Apache-2.0. Open builtins included here must follow the same license. Product-specific or proprietary builtins and rule packs belong in separate crates and should be injected through `BuiltinsProvider`.
