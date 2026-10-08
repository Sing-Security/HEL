# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-10-07

### Added

- **Arena Allocator**: New `arena` module with arena-based memory allocation for AST nodes
  - `ArenaParser` struct for building arena-allocated ASTs
  - `AstNode<'arena>` enum with lifetime parameter for arena-allocated nodes
  - `evaluate_arena()` function for high-performance expression evaluation
  - `evaluate_with_resolver_arena()` and `evaluate_with_context_arena()` for advanced use cases
  - ArenaParser can be reset and reused for multiple evaluations
- **Performance**: Arena allocation provides significant performance improvements:
  - O(1) bump pointer allocation vs global allocator overhead
  - Improved cache locality with adjacent memory layout
  - Batch deallocation - drop arena to free all nodes at once
- **Feature Flag**: New `arena` feature (enabled by default) for arena allocator
- **Benchmarks**: Added criterion benchmarks comparing heap vs arena allocation performance
  - Parse benchmarks for both approaches
  - Evaluation benchmarks for both approaches
  - Batch evaluation benchmarks simulating rule engine workloads
- `rust-version = "1.85"` is now declared.

### Changed

- **Breaking - the public enums are `#[non_exhaustive]`**: `AstNode`, `Comparator`, `Value`,
  `EvalError`, `ErrorKind`, `FieldType` and `PackageError` now need a wildcard arm when
  matched from another crate. Adding a variant to any of them is no longer a breaking change.
- **`core.contains` uses the language's `==`**: list membership is decided by the same
  comparison the `CONTAINS` operator uses, rather than by a second, separately written
  equality. The two spellings of "is this element in this list" can no longer disagree. As a
  consequence a *list* element is no longer matched recursively - `core.contains([[1, 2]],
  [1, 2])` is now `false`, matching `[[1, 2]] CONTAINS [1, 2]`.

### Fixed

- **The grammar is anchored**: `validate_expression`, `parse_expression` and `parse_script`
  now reject trailing input. pest matches a *prefix*, so an unanchored grammar accepted
  `binary.arch == "elf" garbage` as the valid prefix and silently ignored the rest - syntax
  validation did not actually validate. Rules that previously "validated" while containing
  trailing junk will now correctly fail to parse.
- **The resolver-based evaluators no longer panic**: `evaluate_with_resolver`,
  `evaluate_with_context`, `evaluate_with_trace`, `evaluate_with_resolver_arena` and
  `evaluate_with_context_arena` returned `Result` but reached the parser through the
  panicking `parse_rule`, so malformed input unwound the caller. They now return
  `EvalError::ParseError`, which is what their `# Errors` sections already promised. Only
  input that previously panicked behaves differently. `parse_rule` itself still panics, by
  contract, and says so under `# Panics`.
- **Documentation**: the resolver-based evaluators' `# Errors` sections claimed a missing
  attribute produced `EvalError::UnknownAttribute`; it resolves to `Value::Null` instead, and
  the docs now say so.

### Removed

- `FactsEvalContext::from_json` - unused, and the JSON shape it accepted was never specified.

### Dependencies

- Added `bumpalo` 3.x with `collections` feature for arena allocation

## [0.2.0] - 2026-01-21

### Added

- **Expression Validation API**: New `validate_expression()` function for syntax validation with detailed line/column error information
- **Expression Parsing API**: New `parse_expression()` function to parse expressions into AST for advanced use cases
- **Facts-Based Evaluation**: New `FactsEvalContext` struct providing simple key-value store for facts
- **Simple Evaluation API**: New `evaluate()` function for straightforward expression evaluation
- **Script Support**: New `parse_script()` and `evaluate_script()` functions for `.hel` script files
- **Let Bindings**: Full support for reusable sub-expressions in scripts via `let` keyword
- **Enhanced Error Type**: New `HelError` type with optional line/column parse error information
- **Error Classification**: New `ErrorKind` enum for categorizing errors (ParseError, EvaluationError, TypeError, UnknownAttribute)
- **Value Conversions**: Implemented `From` traits for `&str`, `String`, `bool`, `f64`, `i32`, and `u64` for ergonomic `Value` creation
- **Script AST**: New `Script` type representing parsed `.hel` files with let bindings and final expression
- **Expression Type Alias**: New `Expression` type alias for `AstNode` to clarify API usage

### Changed

- **Version**: Bumped to 0.2.0 following semantic versioning (new features, backward compatible)
- **Documentation**: Significantly improved README with Quick Start guide and Rule Engine integration examples
- **Documentation**: Added comprehensive rustdoc comments to all new public APIs with usage examples

### Fixed

- **Boolean Expression Evaluation**: Improved handling of boolean expressions (Comparison, And, Or) in value evaluation context
- **Variable Resolution**: Added proper variable lookup in identifier evaluation for let bindings
- **Multi-line Script Parsing**: Enhanced script parser to handle multi-line expressions and proper expression boundary detection

## [0.1.1] - 2026-01-20

### Initial Release

- Core HEL expression language parser using Pest grammar
- AST representation with support for:
  - Boolean literals and expressions (AND, OR)
  - String, Number, and Float literals
  - Attribute access (object.field notation)
  - Comparisons (==, !=, >, >=, <, <=, CONTAINS, IN)
  - List and Map literals
  - Function calls with namespace support
- `HelResolver` trait for custom attribute resolution
- `EvalContext` for evaluation with resolver and built-ins
- Built-in function registry system:
  - `BuiltinsProvider` trait for domain-specific functions
  - `BuiltinsRegistry` for namespace-aware function dispatch
  - `CoreBuiltinsProvider` with standard functions (len, contains, upper, lower)
- Evaluation tracing for audit trails:
  - `evaluate_with_trace()` function
  - `EvalTrace` and `AtomTrace` types for detailed comparison tracking
  - Deterministic fact usage tracking
- Schema and package system:
  - Schema parser for `.hel` schema files
  - Package manifest support (`hel-package.toml`)
  - `PackageRegistry` for loading and resolving packages
  - Type environment building with collision detection
- Pure, deterministic evaluation (stable maps, no global state)
- Apache-2.0 license

[0.3.0]: https://github.com/Sing-Security/HEL/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Sing-Security/HEL/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Sing-Security/HEL/releases/tag/v0.1.1
