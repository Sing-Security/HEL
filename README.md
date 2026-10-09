# HEL - Heuristic Expression Language

SPDX-License-Identifier: Apache-2.0

## Overview

- HEL (Heuristic Expression Language) is a small, deterministic, auditable expression language and its reference implementation.
- This crate provides a pest-based parser, a compact typed AST, deterministic evaluators, a pluggable built-ins registry, schema and package loaders for domain types, and a trace facility that produces stable, auditable evaluation traces.
- The crate is deliberately domain-agnostic. Registers, rule packs and built-ins that know about a particular kind of data belong in separate crates, injected at runtime through the built-ins provider interface.

## Quick Start

HEL provides a simple, high-level API for expression validation and evaluation:

### Basic Expression Validation

```rust
use hel::validate_expression;

// Validate syntax without evaluation
let expr = r#"binary.arch == "x86_64" AND security.nx == false"#;
assert!(validate_expression(expr).is_ok());  // Err carries a line and column
```

### Expression Evaluation with Facts

```rust
use hel::{evaluate, FactsEvalContext, Value};

// Create evaluation context with facts
let mut ctx = FactsEvalContext::new();
ctx.add_fact("binary.arch", Value::String("x86_64".into()));
ctx.add_fact("security.nx", Value::Bool(false));

// Evaluate expression
let expr = r#"binary.arch == "x86_64" AND security.nx == false"#;
let result = evaluate(expr, &ctx).expect("evaluation failed");  // true
assert!(result);
```

### Script Files with Let Bindings

HEL supports `.hel` script files with reusable let bindings:

```rust
use hel::{evaluate_script, FactsEvalContext, Value};

let mut ctx = FactsEvalContext::new();
ctx.add_fact("manifest.permissions", Value::List(vec![
    Value::String("READ_SMS".into()),
    Value::String("SEND_SMS".into()),
]));
ctx.add_fact("binary.entropy", Value::Number(8.0));

let script = r#"
    let has_sms_perms = 
      manifest.permissions CONTAINS "READ_SMS" AND
      manifest.permissions CONTAINS "SEND_SMS"
    
    let has_obfuscation = binary.entropy > 7.5
    
    has_sms_perms AND has_obfuscation
"#;

let result = evaluate_script(script, &ctx).expect("evaluation failed");  // true
assert!(result);
```

### Arena Allocation for High Performance

For high-throughput scenarios (e.g., evaluating hundreds of rules per request), HEL provides an arena allocator that dramatically improves performance by reducing allocation overhead and improving cache locality:

```rust
# #[cfg(feature = "arena")] {
use hel::arena::{ArenaParser, evaluate_arena};
use hel::{FactsEvalContext, Value};

let mut ctx = FactsEvalContext::new();
ctx.add_fact("binary.arch", Value::String("x86_64".into()));
ctx.add_fact("security.nx", Value::Bool(false));

// One parser, reused across evaluations.
let parser = ArenaParser::new();

let expr = r#"binary.arch == "x86_64" AND security.nx == false"#;
let result = evaluate_arena(expr, &ctx, &parser).expect("evaluation failed");  // true
assert!(result);
# }
```

**When to use arena allocation:**
- Evaluating many expressions in a tight loop (e.g., forward-chaining rule engines)
- Expression lifetime is known and bounded (e.g., single request processing)
- Memory pressure from many small heap allocations is a concern

**Performance benefits:**
- **Faster allocation**: O(1) bump pointer allocation vs global allocator overhead
- **Better cache locality**: AST nodes are adjacent in memory
- **Batch deallocation**: Dropping the arena frees all nodes at once

**Example: Reusing arena for multiple evaluations:**

```rust
# #[cfg(feature = "arena")] {
use hel::arena::{ArenaParser, evaluate_arena};
use hel::{FactsEvalContext, Value};

let mut ctx = FactsEvalContext::new();
ctx.add_fact("data.x", Value::Number(42.0));

let mut parser = ArenaParser::new();

let result1 = evaluate_arena(r#"data.x == 42"#, &ctx, &parser).expect("eval failed");
assert!(result1);

parser.reset();

let result2 = evaluate_arena(r#"data.x > 0"#, &ctx, &parser).expect("eval failed");
assert!(result2);
# }
```

## Goals
- Determinism: evaluation order and iteration are stable (stable maps, deterministic traces).
- Auditability: fine-grained atom-level traces that show resolved inputs and atom results.
- Extensibility: runtime injection of domain built-ins via a clear provider/registry API.
- Minimal surface area: provide primitives (parser, AST, evaluator, trace, schema loader) rather than a monolithic runtime.

## What this crate provides (public capabilities)

### Expression Validation and Parsing
- **Expression Validation**: `validate_expression(expr: &str) -> Result<(), HelError>` - validate syntax without evaluation
- **Expression Parsing**: `parse_expression(expr: &str) -> Result<Expression, HelError>` - parse into AST
- **Script Parsing**: `parse_script(script: &str) -> Result<Script, HelError>` - parse `.hel` files with let bindings

### Expression Evaluation
- **Simple Evaluation**: `evaluate(expr: &str, context: &FactsEvalContext) -> Result<bool, HelError>` - evaluate with facts
- **Script Evaluation**: `evaluate_script(script: &str, context: &FactsEvalContext) -> Result<bool, HelError>` - evaluate scripts with let bindings
- **Advanced Evaluation**: Resolver-based evaluation via `evaluate_with_resolver()` and `evaluate_with_context()`
- **Arena Evaluation**: `arena::evaluate_arena(expr: &str, context: &FactsEvalContext, parser: &ArenaParser)` - high-performance arena-allocated evaluation

### Context and Data
- **FactsEvalContext**: Simple key-value store for facts (e.g., "binary.arch" -> "x86_64")
- **HelResolver** trait: Custom attribute resolution for advanced integrations
- **Value** type: `Null`, `Bool`, `String`, `Number`, `List`, `Map`

### Error Handling
- **HelError**: Enhanced error type with line/column information for parse errors
- **EvalError**: Evaluation-time errors (type mismatches, unknown attributes, etc.)
- Clear error messages for common mistakes

### Low-level APIs
- **Direct Parsing**: `parse_rule(condition: &str) -> AstNode` - parse straight to an AST, panicking on bad input
- **AST**: `AstNode` variants: `Bool`, `String`, `Number`, `Float`, `Identifier`, `Attribute`, `Comparison`, `And`, `Or`, `ListLiteral`, `MapLiteral`, `FunctionCall`
- **Comparators**: `==`, `!=`, `>`, `>=`, `<`, `<=`, `CONTAINS`, `IN`

### Builtins and Extensibility
- `BuiltinsProvider` trait and `BuiltinsRegistry` for namespace-aware function dispatch
- `BuiltinFn` type: pure, deterministic functions that map argument `Value`s to a `Result<Value, EvalError>`
- `CoreBuiltinsProvider` included with generic functions (`core.len`, `core.contains`, `core.upper`, `core.lower`, `core.is_null`)

### Trace & Audit
- `evaluate_with_trace(condition, resolver, Option<&BuiltinsRegistry>) -> Result<EvalTrace, EvalError>`
- `EvalTrace` contains deterministic list of `AtomTrace` entries and sorted list of `facts_used()`
- Pretty-print helpers for deterministic, human-readable traces

### Schema and Package System
- Schema parser and in-memory `Schema` representation (`FieldType`, `TypeDef`, `FieldDef`)
- Package manifest type `PackageManifest` (`hel-package.toml`), `SchemaPackage`, and `PackageRegistry`
- Deterministic package resolution and type merging with collision detection

## Integration with Rule Engines

HEL is designed to be embedded in rule engines and security analysis tools. Here's how to integrate HEL into your application:

### Example: Malware Detection Rule Engine

```rust
use hel::{evaluate_script, FactsEvalContext, Value};
use std::fs;

struct MalwareRule {
    name: String,
    description: String,
    script_path: String,
}

fn check_sample(sample: &BinarySample, rules: &[MalwareRule]) -> Vec<String> {
    // Build facts from sample
    let mut ctx = FactsEvalContext::new();
    ctx.add_fact("binary.arch", Value::String(sample.arch.clone().into()));
    ctx.add_fact("binary.entropy", Value::Number(sample.entropy));
    ctx.add_fact("manifest.permissions", Value::List(
        sample.permissions.iter()
            .map(|p| Value::String(p.clone().into()))
            .collect()
    ));
    ctx.add_fact("strings.count", Value::Number(sample.string_count as f64));
    
    // Evaluate all rules
    let mut detections = Vec::new();
    for rule in rules {
        // Load and evaluate .hel script
        let script = fs::read_to_string(&rule.script_path)
            .expect("Failed to load rule");
        
        match evaluate_script(&script, &ctx) {
            Ok(true) => {
                println!("✓ Rule matched: {}", rule.name);
                detections.push(rule.name.clone());
            }
            Ok(false) => {
                println!("  Rule did not match: {}", rule.name);
            }
            Err(e) => {
                eprintln!("✗ Rule evaluation error in {}: {}", rule.name, e);
            }
        }
    }
    
    detections
}

struct BinarySample {
    arch: String,
    entropy: f64,
    permissions: Vec<String>,
    string_count: usize,
}
```

### Example Rule File: `android-malware.hel`

```hel
# Check for suspicious SMS permissions
let has_sms_perms = 
  manifest.permissions CONTAINS "READ_SMS" AND
  manifest.permissions CONTAINS "SEND_SMS"

# Check for code obfuscation indicators
let has_obfuscation = 
  binary.entropy > 7.5 OR
  strings.count < 10

# Final detection logic
has_sms_perms AND has_obfuscation
```

### Best Practices for Integration

1. **Validation Before Deployment**: Always validate rule scripts before loading them.
   A script is not a single expression - it has `let` bindings - so use `parse_script`,
   not `validate_expression`, to check one:

   ```rust
   use std::fs;

   fn load_rule(path: &str) -> Result<hel::Script, Box<dyn std::error::Error>> {
       let script = fs::read_to_string(path)?;
       Ok(hel::parse_script(&script)?)  // Catch syntax errors before deployment
   }
   ```

2. **Error Handling**: Distinguish between parse errors (rule bugs) and evaluation errors (data issues):

   ```rust
   use hel::{evaluate_script, ErrorKind, FactsEvalContext};

   fn check(script: &str, ctx: &FactsEvalContext) {
       match evaluate_script(script, ctx) {
           Ok(result) => { /* process result */ }
           Err(e) if matches!(e.kind, ErrorKind::ParseError) => {
               eprintln!("Rule has syntax error: {}", e);
           }
           Err(e) => {
               eprintln!("Evaluation error: {}", e);
           }
       }
   }
   ```

3. **Validation at load time**: parse each rule once when it is loaded, so a syntax error
   surfaces at startup rather than on the first evaluation:

   ```rust
   use hel::{parse_script, Script};

   fn load_rules(sources: &[String]) -> Result<Vec<Script>, hel::HelError> {
       sources.iter().map(|s| parse_script(s)).collect()
   }
   ```

   Note that evaluation takes the script text, not a `Script` - `evaluate_script` and
   `evaluate` re-parse on each call. Parsing up front is still worth doing to fail fast;
   it does not by itself make repeated evaluation cheaper. For a tight loop, the arena
   allocator (`hel::arena::ArenaParser`, reused and reset between calls) is the lever the
   crate currently offers.

## Advanced Usage Examples

- Parse an expression into an AST:

```rust
use hel::parse_rule;

let ast = parse_rule(r#"binary.format == "elf" AND security.nx_enabled == true"#);
// `ast` is an `AstNode` representing the parsed expression
```

- Evaluate with a simple resolver:

```rust
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
let result = evaluate_with_resolver(r#"binary.format == "elf""#, &resolver)
    .expect("evaluation failed");
assert!(result);
```

- Evaluate with builtins and capture a trace:

```rust
use hel::builtins::{BuiltinsRegistry, CoreBuiltinsProvider};
use hel::{evaluate_with_trace, HelResolver, Value};

struct MyResolver;
impl HelResolver for MyResolver {
    fn resolve_attr(&self, _: &str, _: &str) -> Option<Value> { None }
}

let mut registry = BuiltinsRegistry::new();
registry.register(&CoreBuiltinsProvider).expect("registration failed");

let trace = evaluate_with_trace("core.len([1,2,3]) == 3", &MyResolver, Some(&registry))
    .expect("trace failed");
assert!(trace.result);
println!("{}", trace.pretty_print()); // deterministic, human-friendly audit trail
```

## Requirements

- Rust 1.85 or newer (the floor comes from the `toml` dependency tree, not from the language
  features used here).
- No Cargo features are required. `arena` is on by default; `default-features = false` drops
  the arena module and the `bumpalo` dependency with it.

## Design notes

- **Determinism**
  - Internal maps use `BTreeMap` and lists are iterated stably, so results do not vary between runs.
  - Traces and `facts_used()` are sorted, so audit logs are stable.
- **Pure built-ins**
  - Built-ins must be pure and deterministic; they must not perform unbounded I/O or rely on global mutable state. The registry enforces namespace isolation and stable ordering.
- **Error handling**
  - The low-level resolver-based evaluators return `Result<_, EvalError>`; the high-level entry points (`evaluate`, `evaluate_script`, `evaluate_arena`) return `Result<_, HelError>`. `EvalError` covers parse errors, type mismatches, unknown attributes and invalid operations; `HelError` adds line and column information and a coarse `ErrorKind`.
- **Limits**
  - The language is declarative: comparisons, `AND`/`OR`, literals, attribute access and function calls. There is no arithmetic (`+`, `-`, `*`, `/`), no negation, and no control flow.
  - Function calls require a `BuiltinsRegistry` in the evaluation context. Without one they fail with an `InvalidOperation` error rather than silently evaluating to false.
  - The crate exposes primitives - parser, AST, evaluators, trace, schema loader - and deliberately does not provide a monolithic compiler or a rule engine for any particular domain.
- **Numbers**
  - Runtime numbers are `f64`. Integer literals are stored as `u64` in the AST and converted on use, so integers above 2^53 lose precision.
  - There is no regex engine here. If a custom built-in pattern-matches, it is responsible for keeping that bounded and deterministic.

## Documentation and where to look next

- `hel::schema` - package manifests, `SchemaPackage`, schema parsing helpers.
- `hel::builtins` - provider/registry API and `CoreBuiltinsProvider`.
- `hel::trace` - trace capture and pretty-print helpers.
- `src/lib.rs` - the parser entry points and the AST.
- The tests in `src/*` are the specification for edge-case behaviour: NaN handling, built-in dispatch, trace ordering, package collision detection.

## Contributing

- Preserve determinism and auditability.
- Keep the built-ins shipped here generic and domain-agnostic.
- When a change affects evaluation semantics, add a deterministic test for it.
- Avoid `unsafe` in public APIs unless it is strictly necessary and documented.

## License

Apache-2.0. Built-ins shipped with this crate are under the same licence; built-ins that belong to a particular domain or product belong in separate crates and should be injected through `BuiltinsProvider`.
