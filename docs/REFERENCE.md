# HEL — Quick Reference

This document is a concise, developer-focused reference for the HEL (Hermes Expression Language) crate. Use this file when you need quick lookups for language primitives, important crate entry points, schema/package hints, trace shapes, and extension points. For the full language specification and rationale see `docs/hel.md`. For integration examples see `docs/USAGE.md` and the crate `README.md`.

Contents
- Language Quick Reference
  - Types
  - Literals
  - Operators & precedence
  - Expressions & control flow
  - Collections API (common)
  - Option/Result helpers
  - Builtins (summary)
- Crate Entry Points (public surface)
- Schema & package quick facts
- Trace & audit quick facts
- Builtins provider & registry (summary)
- Best practices (short)
- Examples (compact)

-- Language Quick Reference

Types
- Primitives:
  - `bool` — boolean
  - `int` — signed 64-bit integer (legacy; prefer `Number` if adopted)
  - `float` / `Number` — IEEE-754 64-bit floating point (Hel uses `Number` in spec)
  - `string` — UTF-8 string
  - `bytes` — immutable byte array
  - `time` — UTC instant
- Collections:
  - `list<T>` — ordered sequence
  - `map<string, T>` — mapping with string keys (deterministic ordering internally)
- Algebraic:
  - `option<T>` — `none` | `some(T)`
  - `result<T, E>` — `ok(T)` | `err(E)`

Literals
- Booleans: `true`, `false`
- Integers: `0`, `-42`, `1_000_000`
- Floats: `3.14`, `-0.5`, `1.0e-6`
- Strings: `"text"` — escape: `\n`, `\t`, `\"`, `\\`
- Bytes: `0xDEADBEEF`, and helper form like `hex("deadbeef")` if provided by builtins
- Lists: `[1, 2, 3]`
- Maps: `{ "k": 1, "v": 2 }`
- Option/result: `none`, `some(expr)`, `ok(expr)`, `err(expr)`
- Regex: `re("[A-Z]{2}\\d+", flags="i")` — if supported by builtin constructors
- Time: `time("2025-10-01T12:34:56Z")` — if time type is enabled

Operators & Precedence (high → low)
1. `!` (logical NOT)
2. `*`, `/`, `%`
3. `+`, `-`
4. Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
5. `&&`
6. `||`
7. `??` (null-coalescing)
8. `|>` (pipeline / left-to-right function application)

Parentheses `()` override precedence. Evaluation is eager (strict) and left-to-right, with short-circuiting semantics for `&&` and `||`.

Common Expressions & Control Flow
- `if cond then a else b` — expression-level conditional
- `match expr { pat => expr, ... }` — simple pattern matching with literal and wildcard `_`
- `let name = expr; expr2` — let-bindings inside expressions (limited scope)
- Safe navigation: `?.` — `pkg.publisher?.name` yields `none` if missing
- Null-coalescing: `a ?? b` — returns `b` if `a` is `none`

Collections API (typical builtins)
- Lists:
  - `.len`, `.is_empty`
  - `.map(f)`, `.filter(f)`, `.reduce(init, f)`
  - `.any(f)`, `.all(f)`, `.contains(x)`
  - `.unique()`, `.concat(l2)`, `.take(n)`, `.drop(n)`, `.sort_by(f)` (stable)
- Maps:
  - `.keys()`, `.values()`
  - `.get(k) -> option<T>`
  - `.has(k)` or `CONTAINS` (key membership)
  - `.merge(m2)`

Option / Result helpers
- `option<T>`:
  - `.is_some`, `.is_none`, `.unwrap_or(default)`, `.ok` / `.err` as accessors
- `result<T, E>`:
  - `.is_ok`, `.is_err`, `.ok`, `.err`, `.unwrap_or(default)` (returns default instead of panicking)

Builtins (summary)
- Strings: `str.len`, `str.contains`, `str.starts_with`, `str.ends_with`, `str.lower`, `str.upper`, `str.trim`, `str.replace(a,b)`
- Regex helpers (open): `regex.is_match(re, s)`, `regex.find_all(re, s) -> list<string>`
- Bytes: `entropy(bytes) -> Number`, `bytes.len`, `bytes.slice(start, len)`, `hash.sha256(bytes) -> bytes`, `hash.sha1`, `hash.md5` (discouraged)
- Collections helpers: `list.unique()`, `list.concat()`, `list.take()`, `list.drop()`
- IP/CIDR: `ip.parse(string) -> result<ip, string>`, `cidr.parse(string)`, `ip.in_cidr(ip, cidr) -> bool`
- Time: `now()` — feature-gated; host must inject deterministic clock in production

Note: The set of builtins available at runtime depends on the `BuiltinsRegistry` and any closed `BuiltinsProvider`s attached by the host.

-- Crate Entry Points (public surface)
These are the main modules and types you will use when embedding `hel`:

- Parser
  - `HelParser` — pest-generated parser type (parses HEL source into AST or returns parse errors with spans)
  - Parsing helpers: `HelParser::parse_expression(...)` or similar (check the crate exports)

- Schema (package loaders)
  - `hel::schema::SchemaPackage` — loader for `hel-package.toml` + `.hel` files
  - `hel::schema::Schema` — merged schema used by resolvers and validation
  - Helpers: `SchemaPackage::from_directory(...)`, `schema.validate()`, `schema.merge()`

- Evaluation & Tracing
  - `hel::trace::evaluate_with_trace(ast, resolver, registry)` — evaluate an AST and return `(Value, EvalTrace)` or error
  - `hel::trace::EvalTrace` — trace representation produced by evaluator

- Builtins
  - `hel::builtins::BuiltinsProvider` — trait for host-provided closed builtins
  - `hel::builtins::BuiltinsRegistry` — registry that holds builtin implementations and metadata
  - `hel::builtins::CoreBuiltinsProvider` — default open builtins provider shipped with the crate
  - `hel::builtins::BuiltinFn` — function signature type (accepts `Value` args and returns `Result<Value, BuiltinError>`)

- Public data fact types (domain helpers)
  - Examples declared in `src/lib.rs` (may include): `BinaryInfo`, `SecurityFlags`, `SectionInfo`, `ImportInfo`, `TaintFlow`, `FunctionCall`, `MemoryOperation`, `SymQueryRequest`
  - These are convenience types used by product layers; do not hard-code product logic in hel

- Resolver interface
  - Programmatic `HelResolver` trait — maps attribute paths to runtime `Value` objects

-- Schema & Package Quick Facts
- Manifest: `hel-package.toml`
  - Required fields: `name`, `version` (semver), `schema_files` (ordered list), optional `authors`, `description`
- Schema files: `.hel` — declare `type` definitions (primitives, `List<T>`, `Map<String, T>`, optional fields `?`)
- Loading:
  - Use `SchemaPackage::from_directory(dir)` to load; then `pkg.merge()` to obtain `Schema`
  - Validation MUST ensure type references resolve and duplicate names are rejected
- Versioning:
  - Every schema package must be versioned. Hosts must record `name@version` for each evaluation.

-- Trace & Audit Quick Facts
- Use `evaluate_with_trace` to obtain both result and a deterministic `EvalTrace`.
- Trace must include:
  - Expression identifier (hash), expression text (or secure pointer), schema snapshots, builtin registry snapshot
  - Ordered steps: literal eval, identifier resolve, builtin_call (must include builtin name + version), operation, errors, eval_end
  - Value summaries: primitives inline, large blobs replaced with `sha256` + length
- Persist with: expression text, schema manifests/hashes, registry snapshot, resolver input snapshot, and the `EvalTrace` itself
- Canonical serialization: deterministic JSON (sorted keys, consistent float formatting) — include canonicalization id in trace meta

-- Builtins Provider & Registry (short)
- Open builtins live in crate `CoreBuiltinsProvider` and are safe, well-documented, and deterministic.
- Closed builtins:
  - Implement `BuiltinsProvider` in a separate (closed) crate.
  - Register with `BuiltinsRegistry` at host startup: `registry.register_provider(Box::new(MyProvider::new(...)))`
  - Each builtin must carry metadata: `name`, `version`, `description`, and determinism guarantee
- Conflict resolution:
  - Registry policy is deterministic. Prefer explicit host ordering; warn on collisions; first-registered wins unless configured otherwise.
- Builtins must:
  - Return `Result<Value, BuiltinError>`; never panic
  - Be bounded in CPU/memory; enforce input limits for regex and large data
  - Emit trace metadata for audit (builtin_version, inputs summary, output summary, duration_ms)

-- Best Practices (very short)
- Determinism:
  - Resolver must be side-effect-free and deterministic.
  - Builtins that require time or randomness must accept deterministic injection points.
  - Use `BTreeMap` internally for maps recorded in trace.
- Auditability:
  - Always store expression source (or hash), schema package versions, and registry snapshot alongside the trace.
  - Persist traces in canonical JSON format.
- Safety:
  - Avoid exposing `unsafe` in public crate APIs. Wrap and document `unsafe` if used internally.
  - Validate sizes for untrusted inputs; return errors for oversized payloads.
- Extensibility:
  - Put product-specific logic in closed crates. Use `BuiltinsProvider` to attach proprietary functions.
  - Author schema packages and version them; do not embed product schema into the open crate.

-- Examples (compact)
- Parse & evaluate (concept)
```text
// 1) Parse source -> AST
let ast = HelParser::parse_expression("entropy(file.bytes) > 7.0")?;
//
// 2) Load schema and build resolver
let pkg = SchemaPackage::from_directory("products/Desmond/schema")?;
let schema = pkg.merge()?;
let resolver = SchemaBackedResolver::new(&schema, &host_facts);
//
// 3) Build builtin registry and attach closed provider if needed
let mut registry = BuiltinsRegistry::default();
registry.register_provider(Box::new(CoreBuiltinsProvider::default()));
registry.register_provider(Box::new(MyProductBuiltins::new(...)));
//
// 4) Evaluate with trace
let (value, trace) = evaluate_with_trace(&ast, &resolver, &registry)?;
```

- Pattern: safe navigation + default
```text
(pkg.publisher?.name ?? "unknown") == "Acme Corp"
```

- Collections example
```text
files
  .filter(f => f.size > 0)
  .map(f => f.hash.sha256)
  .unique()
  .contains(expected_hash)
```

-- Where to read more
- Full language spec: `docs/hel.md`
- HEL crate README: `README.md`
- `binx` HEL integration notes: `docs/binx/HEL.md`
- Developer docs: `docs/` (USAGE.md, SCHEMA.md, BUILTINS.md, TRACING.md, CONTRIBUTING.md)

-- Quick reference checklist before shipping an integration
- [ ] Expression text and ID recorded
- [ ] Schema package name@version recorded
- [ ] Builtins registry snapshot recorded
- [ ] EvalTrace persisted (canonical JSON)
- [ ] Resolver deterministic and free of I/O
- [ ] Closed builtins implemented in separate crate and injected via `BuiltinsProvider`
- [ ] Tests include trace-level assertions and deterministic fixtures

This reference is intentionally compact. For design rationale, formal language semantics, exhaustive operator behavior, and examples, use `docs/hel.md` and the other documents in this crate's `docs/` directory. If you want, I can generate a small example Rust test harness illustrating `HelParser`, a minimal `HelResolver`, a `BuiltinsRegistry` with `CoreBuiltinsProvider`, and an `evaluate_with_trace` assertion next.