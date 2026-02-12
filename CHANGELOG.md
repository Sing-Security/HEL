# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

## [0.1.1] - 2025-XX-XX

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

[0.2.0]: https://github.com/Sing-Security/hel/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Sing-Security/hel/releases/tag/v0.1.1
