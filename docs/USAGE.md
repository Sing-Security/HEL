desmond/forgecore/crates/hel/docs/USAGE.md#L1-400
# HEL — Quickstart & Usage Guide

This quickstart is intended to get you productive with the `hel` crate quickly. It focuses on practical integration patterns you will use most often when embedding HEL in a host product:

- parsing and validating expressions,
- wiring a resolver (programmatic or schema-driven),
- injecting builtins (open and closed),
- evaluating expressions with a deterministic trace suitable for audits,
- handling errors and tests.

This document is conceptual: prefer the crate's exported API (`HelParser`, `evaluate_with_trace`, `hel::schema` loaders, and `BuiltinsProvider`) rather than undocumented helpers. Keep product logic out of the open crate — implement product-specific builtins in closed crates and inject them via the provider interface.

---

## 1 — Files & entry points

Important files and symbols you will use:

- `README.md` (crate): introductory surface and design constraints.
- `docs/hel.md`: authoritative language specification.
- `hel::schema`: schema package loaders and helpers (`SchemaPackage`, `Schema`).
- `hel::builtins`: builtin provider/registry interfaces (`BuiltinsProvider`, `BuiltinsRegistry`, `BuiltinFn`).
- `hel::trace`: tracing helpers and `evaluate_with_trace`.

See `docs/hel.md` for the language reference and operator semantics.

---

## 2 — Parse an expression

Start by parsing an expression to produce an AST and to validate syntax. This is usually the first step before type-checking or evaluation.

Example HEL source (literal expression):

```/dev/null/example.hel#L1-6
status == "signed" && entropy(file.bytes) < 7.5
```

Conceptual Rust-style pseudo-flow (do not rely on types not in the crate surface — this is a pattern):

```desmond/forgecore/crates/hel/src/lib.rs#L1-120
// 1) Read expression text (from DB, file, or inline).
let src = r#"status == "signed" && entropy(file.bytes) < 7.5"#;

// 2) Parse: produce AST or structured parse error with spans.
let parse_result = HelParser::parse_expression(src);
// parse_result -> Result<Ast, ParseError>
```

Parse errors include structured span information; record the source text and span when surfacing the error to users or auditors.

---

## 3 — Prepare a resolver (host data)

HEL evaluates expressions against an immutable resolver that maps attribute paths to runtime `Value`s. There are two common approaches:

- Programmatic resolver: implement the `HelResolver` (or equivalent) trait in Rust and map host domain types to HEL `Value`s.
- Schema-based resolver: load a `SchemaPackage` from a directory (manifest + types) and use the schema to validate and guide resolution.

Example schematic resolver signature:

```desmond/forgecore/crates/hel/src/lib.rs#L121-200
pub trait HelResolver {
    /// Resolve an attribute path like `binary.arch` to a HEL Value.
    /// Returns `None` for missing attributes (treated as `null`).
    fn resolve_attr(&self, object: &str, field: &str) -> Option<Value>;
}
```

Design notes:
- Missing attributes return `None` → interpreted as `null` per HEL semantics.
- Keep resolvers deterministic (no I/O during resolve calls).
- For large hosts, prefer a resolver that performs a single pre-serialization pass into a stable `Value` map you can feed to the engine.

---

## 4 — Builtins: open vs closed

`hel` exposes an extension point for builtins. The common pattern is:

1. Use the open builtin registry for generic helpers (strings, lists, hashing, entropy).
2. Implement closed/product builtins in a separate crate and provide them via a `BuiltinsProvider` implementation at runtime.

Conceptual builtin injection:

```desmond/forgecore/crates/hel/src/lib.rs#L201-320
let mut registry = BuiltinsRegistry::default();
// Register open builtins provided by this crate:
registry.register_core_builtins(CoreBuiltinsProvider::default());

// Host injects closed builtins:
let closed_provider = MyProductBuiltins::new(...);
registry.register_provider(Box::new(closed_provider));
```

Rules:
- Closed builtins must be deterministic and documented.
- Avoid side-effects and keep evaluation pure from the engine's perspective.

---

## 5 — Evaluate with trace (audit-friendly)

Use `evaluate_with_trace` (or equivalent) to evaluate an expression while collecting a structured trace. The trace must include: expression spans, intermediate values, builtin calls and their inputs/outputs, and final result.

High-level evaluation flow:

```desmond/forgecore/crates/hel/src/lib.rs#L321-420
// evaluator inputs:
// - parsed AST
// - resolver (immutable, deterministic)
// - builtin registry/provider
// returns:
// - Result<Value, EvalError>
// - EvalTrace (detailed step-by-step)
let (value, trace) = evaluate_with_trace(&ast, &resolver, &registry)?;
```

Audit requirements:
- Persist the expression text (or its canonical hash) and the `Schema` version(s) used.
- Persist the builtin registry version (or list of builtin names + versions).
- Persist the emitted `EvalTrace` with stable serialization.

---

## 6 — Error handling & safety

- The public APIs return `Result<T, Error>`; do not rely on panics.
- Parse errors contain spans and human-friendly messages.
- Type and runtime errors are explicit; prefer returning `result<T, E>` from builtins that can fail.
- Numeric overflow or invalid ops return structured errors, not panics.

Example: handling an eval outcome:

```desmond/forgecore/crates/hel/src/lib.rs#L421-520
match evaluate_with_trace(&ast, &resolver, &registry) {
    Ok((val, trace)) => {
        // Record result and trace for audit.
    }
    Err(err) => {
        // Persist parse/type/eval error with source text and spans.
    }
}
```

---

## 7 — Testing recommendations

- Unit test expressions with fixed, deterministic resolvers (prefer in-memory resolvers).
- Include trace assertions where possible: verify key builtin calls and intermediate values appear in the trace.
- When evaluating floating-point values, assert within an explicit epsilon or use `approx_eq`.
- For time-dependent builtins like `now()`, ensure the host pins the clock or uses feature-gated deterministic clock injection for tests.

Example test scaffold:

```desmond/forgecore/crates/hel/src/lib.rs#L521-640
#[test]
fn test_entropy_rule() {
    // -- Setup & Fixtures
    let src = "entropy(file.bytes) > 6.0";
    let ast = HelParser::parse_expression(src).unwrap();

    let resolver = InMemoryResolver::from_map(...);
    let registry = BuiltinsRegistry::with_core();

    // -- Exec
    let (val, trace) = evaluate_with_trace(&ast, &resolver, &registry).unwrap();

    // -- Check
    assert!(val.is_bool() && val.as_bool() == true);
    // Optionally assert the trace includes expected builtin call and value.
}
```

Follow the Rust10x testing patterns: structured test sections and deterministic fixtures.

---

## 8 — Packaging expressions & schema

- Version expression bundles (text + metadata) and store canonical identifiers (hash + version).
- Schema packages follow `hel-package.toml` and are loaded via `hel::schema::SchemaPackage::from_directory`.
- Host products must record the Schema package versions used during evaluation.

---

## 9 — Troubleshooting & guidance

- If you see nondeterministic results, verify:
  - The resolver is deterministic and free of I/O.
  - Closed builtins are deterministic for identical inputs.
  - Time or randomness are not used without injection.
- If regex-based builtins are added, ensure they enforce RE2-like or bounded execution semantics to prevent super-linear behavior.
- Keep product logic outside this crate; inject via builtins.

---

## 10 — Next reading

- `SCHEMA.md` — schema package authoring and loaders.
- `BUILTINS.md` — how to implement and inject open vs closed builtins.
- `TRACING.md` — the `EvalTrace` shape and examples of audit evidence.
- `CONTRIBUTING.md` — GEMINI-aligned contribution rules.
