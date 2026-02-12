# HEL — Heuristics Expression Language

Status: OPEN — Apache-2.0  
SPDX-License-Identifier: Apache-2.0

## Overview

- HEL (Internally Hermes Expression Language) is a small, deterministic, auditable expression language and reference implementation.
- This crate implements the open core: a pest-based parser, a compact typed AST, deterministic evaluator(s), a pluggable builtins registry, schema/package loaders for domain types, and a trace facility that produces stable, auditable evaluation traces.
- The crate is intentionally product-agnostic: domain-specific or proprietary built-ins and rule packs should be implemented and shipped separately and injected at runtime via the builtins provider interface.

## Quick Start

HEL provides a simple, high-level API for expression validation and evaluation:

### Basic Expression Validation

```rust
use hel::validate_expression;

// Validate syntax without evaluation
let expr = r#"binary.arch == "x86_64" AND security.nx == false"#;
validate_expression(expr)?;  // Returns Ok(()) or detailed parse error
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
let result = evaluate(expr, &ctx)?;  // Returns true
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
    # Define reusable sub-expressions
    let has_sms_perms = 
      manifest.permissions CONTAINS "READ_SMS" AND
      manifest.permissions CONTAINS "SEND_SMS"
    
    let has_obfuscation = binary.entropy > 7.5
    
    # Final boolean expression
    has_sms_perms AND has_obfuscation
"#;

let result = evaluate_script(script, &ctx)?;  // Returns true
```

### Arena Allocation for High Performance

For high-throughput scenarios (e.g., evaluating hundreds of rules per request), HEL provides an arena allocator that dramatically improves performance by reducing allocation overhead and improving cache locality:

```rust
use hel::arena::{ArenaParser, evaluate_arena};
use hel::{FactsEvalContext, Value};

let mut ctx = FactsEvalContext::new();
ctx.add_fact("binary.arch", Value::String("x86_64".into()));
ctx.add_fact("security.nx", Value::Bool(false));

// Create arena parser (can be reused across many evaluations)
let parser = ArenaParser::new();

let expr = r#"binary.arch == "x86_64" AND security.nx == false"#;
let result = evaluate_arena(expr, &ctx, &parser)?;  // Returns true
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
use hel::arena::{ArenaParser, evaluate_arena};

let mut parser = ArenaParser::new();

// Evaluate first expression
let result1 = evaluate_arena(expr1, &ctx, &parser)?;

// Reset arena to reuse memory
parser.reset();

// Evaluate second expression (reuses arena memory)
let result2 = evaluate_arena(expr2, &ctx, &parser)?;
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

### Legacy APIs
- **Low-level Parsing**: `parse_rule(condition: &str) -> AstNode` - direct AST construction
- **AST**: `AstNode` variants: `Bool`, `String`, `Number`, `Float`, `Identifier`, `Attribute`, `Comparison`, `And`, `Or`, `ListLiteral`, `MapLiteral`, `FunctionCall`
- **Comparators**: `==`, `!=`, `>`, `>=`, `<`, `<=`, `CONTAINS`, `IN`

### Builtins and Extensibility
- `BuiltinsProvider` trait and `BuiltinsRegistry` for namespace-aware function dispatch
- `BuiltinFn` type: pure, deterministic functions that map argument `Value`s to a `Result<Value, EvalError>`
- `CoreBuiltinsProvider` included with generic functions (`core.len`, `core.contains`, `core.upper`, `core.lower`)

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

1. **Validation Before Deployment**: Always validate rule scripts before loading them:
   ```rust
   let script = fs::read_to_string("rule.hel")?;
   validate_expression(&script)?;  // Catch syntax errors early
   ```

2. **Error Handling**: Distinguish between parse errors (rule bugs) and evaluation errors (data issues):
   ```rust
   match evaluate_script(&script, &ctx) {
       Ok(result) => { /* process result */ }
       Err(e) if matches!(e.kind, ErrorKind::ParseError) => {
           eprintln!("Rule has syntax error: {}", e);
       }
       Err(e) => {
           eprintln!("Evaluation error: {}", e);
       }
   }
   ```

3. **Performance**: Parse scripts once and reuse the AST:
   ```rust
   let parsed = parse_script(&script)?;
   // Store parsed.bindings and parsed.final_expr
   // Reuse for multiple evaluations
   ```

## Advanced Usage Examples
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
