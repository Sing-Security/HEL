//! HEL - Heuristic Expression Language
//!
//! A small, deterministic expression language for rule engines, security analysis and
//! policy evaluation. You hand it a condition and a way to resolve attribute values; it
//! hands back a boolean, and optionally a trace of how it got there.
//!
//! # Features
//!
//! - **Deterministic**: stable evaluation order and sorted iteration, so the same inputs
//!   always produce the same result and the same trace.
//! - **Auditable**: [`trace`] records every atom's resolved inputs and outcome.
//! - **Extensible**: domain-specific functions are injected at runtime through a
//!   namespace-isolated [`builtins`] registry rather than compiled in.
//! - **Scripts**: multi-line `.hel` scripts with reusable `let` bindings.
//! - **Schemas**: optional declarations of domain types, loaded from `.hel` schema files
//!   or `hel-package.toml` packages ([`schema`]).
//!
//! # Quick Start
//!
//! ## Expression Validation
//!
//! ```
//! use hel::validate_expression;
//!
//! // Validate syntax without evaluation
//! let expr = r#"binary.arch == "x86_64" AND security.nx == false"#;
//! assert!(validate_expression(expr).is_ok());
//! ```
//!
//! ## Expression Evaluation
//!
//! ```
//! use hel::{evaluate, FactsEvalContext, Value};
//!
//! let mut ctx = FactsEvalContext::new();
//! ctx.add_fact("binary.arch", Value::String("x86_64".into()));
//! ctx.add_fact("security.nx", Value::Bool(false));
//!
//! let expr = r#"binary.arch == "x86_64" AND security.nx == false"#;
//! let result = evaluate(expr, &ctx).expect("evaluation failed");
//! assert!(result);
//! ```
//!
//! ## Script Evaluation with Let Bindings
//!
//! ```
//! use hel::{evaluate_script, FactsEvalContext, Value};
//!
//! let mut ctx = FactsEvalContext::new();
//! ctx.add_fact("manifest.permissions", Value::List(vec![
//!     Value::String("READ_SMS".into()),
//! ]));
//! ctx.add_fact("binary.entropy", Value::Number(8.0));
//!
//! let script = r#"
//!     let has_sms = manifest.permissions CONTAINS "READ_SMS"
//!     let has_obfuscation = binary.entropy > 7.5
//!     has_sms AND has_obfuscation
//! "#;
//!
//! let result = evaluate_script(script, &ctx).expect("evaluation failed");
//! assert!(result);
//! ```
//!
//! # Where things live
//!
//! - This module - the grammar entry points, the [`AstNode`] tree, [`Value`], the
//!   [`HelResolver`] trait, and the evaluators.
//! - [`builtins`] - the function registry and the generic `core.*` functions.
//! - [`trace`] - per-atom evaluation traces, for explaining why a rule matched.
//! - [`schema`] - optional declarations of a domain's types and packages.
//! - `arena` - an evaluator that allocates its AST in a reusable bump arena
//!   (feature `arena`, on by default).
//!
//! A condition becomes a boolean in three steps: the pest grammar produces a parse tree,
//! that tree is lowered into an [`AstNode`], and the AST is walked against a resolver.
//! Each step is separately reachable - [`validate_expression`] stops after the first,
//! [`parse_expression`] after the second - so a host can check a rule without running it.
//!
//! # Advanced Usage
//!
//! ## Custom Built-in Functions
//!
//! A host adds vocabulary by implementing [`BuiltinsProvider`] and registering it; see
//! that trait for a worked example. Once registered, the functions are callable from any
//! expression evaluated with that registry:
//!
//! ```
//! use hel::{BuiltinsRegistry, CoreBuiltinsProvider, FactsEvalContext, evaluate_with_context};
//!
//! let mut registry = BuiltinsRegistry::new();
//! registry.register(&CoreBuiltinsProvider).expect("registration failed");
//!
//! let ctx = FactsEvalContext::new();
//! assert!(evaluate_with_context(
//!     r#"core.len(["a", "b"]) == 2"#,
//!     &ctx,
//!     &registry,
//! ).expect("evaluation failed"));
//!
//! // Unregistered namespaces and function names fail rather than returning false.
//! assert!(evaluate_with_context("nope.f()", &ctx, &registry).is_err());
//! ```
//!
//! ## Evaluation Tracing
//!
//! ```
//! use hel::{evaluate_with_trace, HelResolver, Value};
//!
//! struct MyResolver;
//! impl HelResolver for MyResolver {
//!     fn resolve_attr(&self, object: &str, field: &str) -> Option<Value> {
//!         match (object, field) {
//!             ("binary", "format") => Some(Value::String("elf".into())),
//!             _ => None,
//!         }
//!     }
//! }
//!
//! let trace = evaluate_with_trace(
//!     r#"binary.format == "elf""#,
//!     &MyResolver,
//!     None
//! ).expect("trace failed");
//!
//! assert!(trace.result);
//! assert_eq!(trace.atoms.len(), 1);
//! ```
//!
//! # Cargo features
//!
//! - `arena` (enabled by default) - adds the `arena` module, an evaluator that allocates AST nodes
//!   in a bump arena and can reuse that memory across evaluations. It is a pure performance
//!   win and removes no API.
//!
//! # Stability
//!
//! The public enums here ([`AstNode`], [`Comparator`], [`Value`], [`EvalError`],
//! [`ErrorKind`], [`FieldType`], [`PackageError`]) are `#[non_exhaustive]`: matching them
//! from another crate needs a `_` arm, which leaves room to add a variant without that
//! being a breaking change. The structs that carry only public data are not, so they can
//! still be built with a literal.
//!
//! With `default-features = false` the crate builds without [`bumpalo`](https://docs.rs/bumpalo)
//! and the `arena` module is absent; the rest of the public API is unchanged, so a caller
//! that does not need arena allocation pays for nothing.
//!
//! # Limits
//!
//! - The language is not Turing-complete. There is no arithmetic (`+`, `-`,
//!   `*`, `/`), no negation, no assignment, and no control flow - a condition is built from
//!   comparisons, `AND`/`OR`, literals, attribute access and function calls.
//! - Numbers are `f64` at evaluation time. Integer literals are held as `u64` in the AST
//!   and converted on use, so integers above 2^53 lose precision.
//! - `NaN` comparisons are false, as in IEEE 754.
//! - Calling a function without a [`builtins`] registry in the evaluation context is an
//!   error, not a silent default.
//! - There is no borrow or evaluator-level recursion limit; an expression is a fixed tree,
//!   so evaluation terminates by construction, but a recursive custom built-in would not.

#![warn(missing_docs)]
#![forbid(unsafe_code)]
// Runs every `rust` block in README.md under `cargo test --doc` without appending the README
// to the rendered API docs - the include only fires while doctests are being collected.
#![cfg_attr(doctest, doc = include_str!("../README.md"))]

use pest::iterators::Pair;
use pest::Parser;
use std::collections::BTreeMap;
use std::sync::Arc;

pub mod schema;
pub use schema::{
    package::{PackageError, PackageManifest, PackageRegistry, SchemaPackage, TypeEnvironment},
    parse_schema, FieldDef, FieldType, Schema, TypeDef,
};

pub mod builtins;
pub use builtins::{BuiltinFn, BuiltinsProvider, BuiltinsRegistry, CoreBuiltinsProvider};

pub mod trace;
pub use trace::{evaluate_with_trace, AtomTrace as TraceAtom, EvalTrace};

#[cfg(feature = "arena")]
pub mod arena;

/// HEL parser generated by Pest, and the rule set it accepts
///
/// Wrapped in its own module so `missing_docs` can be allowed over the derive output:
/// `pest_derive` generates [`Rule`] and [`HelParser::parse`], neither of which can carry a doc
/// comment.
#[allow(missing_docs)]
mod parser {
    use pest_derive::Parser;

    /// Parses HEL expressions according to the `hel.pest` grammar.
    ///
    /// This is the low-level entry point; most callers want
    /// [`validate_expression`](super::validate_expression),
    /// [`parse_expression`](super::parse_expression) or
    /// [`parse_script`](super::parse_script), which return a `Result` rather than
    /// panicking on malformed input.
    #[derive(Parser)]
    #[grammar = "hel.pest"]
    pub struct HelParser;
}

pub use parser::{HelParser, Rule};

/// One node of a parsed HEL expression - one variant per syntactic construct.
///
/// # Examples
///
/// ```
/// use hel::{parse_expression, AstNode};
///
/// let expr = r#"binary.format == "elf""#;
/// let ast = parse_expression(expr).expect("parse failed");
///
/// match ast {
///     AstNode::Comparison { .. } => println!("It's a comparison"),
///     _ => println!("Something else"),
/// }
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum AstNode {
    /// Boolean literal (true or false)
    Bool(bool),
    /// String literal
    String(Arc<str>),
    /// Integer number literal
    Number(u64),
    /// Float number (f64)
    Float(f64),
    /// Identifier (variable name or unqualified reference)
    Identifier(Arc<str>),
    /// Attribute access (object.field notation)
    Attribute {
        /// Object name
        object: Arc<str>,
        /// Field name
        field: Arc<str>,
    },
    /// Comparison expression (left op right)
    Comparison {
        /// Left operand
        left: Box<AstNode>,
        /// Comparison operator
        op: Comparator,
        /// Right operand
        right: Box<AstNode>,
    },
    /// Logical AND expression
    And(Vec<AstNode>),
    /// Logical OR expression
    Or(Vec<AstNode>),
    /// List literal: [1, 2, 3] or ["a", "b"]
    ListLiteral(Vec<AstNode>),
    /// Map literal: {"key": value, ...}
    MapLiteral(Vec<(Arc<str>, AstNode)>),
    /// Function call: namespace.function(args) or function(args)
    FunctionCall {
        /// Namespace (if qualified, e.g., "core" in core.len)
        namespace: Option<Arc<str>>,
        /// Function name
        name: Arc<str>,
        /// Arguments
        args: Vec<AstNode>,
    },
}

/// Comparison operators supported by HEL.
///
/// # Examples
///
/// ```
/// use hel::evaluate;
/// use hel::FactsEvalContext;
/// use hel::Value;
///
/// let mut ctx = FactsEvalContext::new();
///
/// // FactsEvalContext resolves attributes of the form "object.field"
/// ctx.add_fact("vars.x", 10.0.into());
/// ctx.add_fact("vars.y", 20.0.into());
///
/// // Equality: ==
/// assert!(evaluate(r#"vars.x == 10"#, &ctx).unwrap());
///
/// // Less than: <
/// assert!(evaluate(r#"vars.x < vars.y"#, &ctx).unwrap());
///
/// // Contains (for lists and strings)
/// ctx.add_fact("vars.list", Value::List(vec![1.0.into(), 2.0.into()]));
/// assert!(evaluate(r#"vars.list CONTAINS 1"#, &ctx).unwrap());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Comparator {
    /// Equality (==)
    Eq,
    /// Inequality (!=)
    Ne,
    /// Greater than (>)
    Gt,
    /// Greater than or equal (>=)
    Ge,
    /// Less than (<)
    Lt,
    /// Less than or equal (<=)
    Le,
    /// Contains operator (CONTAINS)
    Contains,
    /// IN operator for membership tests (e.g., "a" IN ["a", "b"])
    In,
}

/// A value produced or consumed during evaluation: a literal, a resolved attribute, a function
/// argument or a function result.
///
/// # Examples
///
/// ```
/// use hel::Value;
/// use std::sync::Arc;
///
/// // Create different value types
/// let null_val = Value::Null;
/// let bool_val = Value::Bool(true);
/// let string_val = Value::String(Arc::from("hello"));
/// let number_val = Value::Number(42.5);
/// let list_val = Value::List(vec![Value::Number(1.0), Value::Number(2.0)]);
/// ```
///
/// # Conversions
///
/// The `Value` type implements `From` for common Rust types for convenience:
///
/// ```
/// use hel::Value;
///
/// let s: Value = "hello".into();
/// let b: Value = true.into();
/// let n: Value = 42.5.into();
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Value {
    /// Null value (represents missing or undefined data)
    Null,
    /// Boolean value (true or false)
    Bool(bool),
    /// String value
    String(Arc<str>),
    /// Numeric value (stored as f64)
    Number(f64),
    /// List of values
    List(Vec<Value>),
    /// Map of string keys to values
    Map(BTreeMap<Arc<str>, Value>),
}

/// Supplies attribute values to the evaluator.
///
/// The host implements this so HEL never has to know where its data comes from: the
/// evaluator asks for `object.field` and gets a [`Value`] back.
///
/// # Examples
///
/// ```
/// use hel::{evaluate_with_resolver, HelResolver, Value};
///
/// struct MyResolver;
///
/// impl HelResolver for MyResolver {
///     fn resolve_attr(&self, object: &str, field: &str) -> Option<Value> {
///         match (object, field) {
///             ("binary", "arch") => Some(Value::String("x86_64".into())),
///             _ => None,
///         }
///     }
/// }
///
/// assert!(evaluate_with_resolver(r#"binary.arch == "x86_64""#, &MyResolver).expect("evaluated"));
/// ```
pub trait HelResolver {
    /// Resolve one `object.field` attribute, or `None` if the host has no such value.
    ///
    /// `None` is not an error: the evaluator substitutes [`Value::Null`], so a comparison
    /// against a missing attribute is simply false. Return `Some(Value::Null)` instead if
    /// you need to distinguish "absent" from "null" - HEL does not.
    fn resolve_attr(&self, object: &str, field: &str) -> Option<Value>;
}

/// A resolver plus, optionally, a built-ins registry.
///
/// This is what the resolver-based entry points hand to the evaluator. Most callers never
/// build one directly - [`FactsEvalContext`] and [`evaluate`] cover the common case, and
/// [`evaluate_with_resolver`] / [`evaluate_with_context`] build this for you. Reach for it to
/// configure a context once and pass it around.
///
/// # Examples
///
/// ```
/// use hel::{EvalContext, HelResolver, Value};
///
/// struct MyResolver;
/// impl HelResolver for MyResolver {
///     fn resolve_attr(&self, object: &str, field: &str) -> Option<Value> {
///         match (object, field) {
///             ("binary", "arch") => Some(Value::String("x86_64".into())),
///             _ => None,
///         }
///     }
/// }
///
/// let resolver = MyResolver;
/// let ctx = EvalContext::new(&resolver);
/// ```
pub struct EvalContext<'a> {
    resolver: &'a dyn HelResolver,
    builtins: Option<&'a builtins::BuiltinsRegistry>,
    /// Variable bindings for let expressions (name -> value)
    variables: BTreeMap<Arc<str>, Value>,
}

impl<'a> EvalContext<'a> {
    /// Create a context with just a resolver (no built-ins)
    pub fn new(resolver: &'a dyn HelResolver) -> Self {
        Self {
            resolver,
            builtins: None,
            variables: BTreeMap::new(),
        }
    }

    /// Create a context with both resolver and built-ins registry
    pub fn with_builtins(
        resolver: &'a dyn HelResolver,
        builtins: &'a builtins::BuiltinsRegistry,
    ) -> Self {
        Self {
            resolver,
            builtins: Some(builtins),
            variables: BTreeMap::new(),
        }
    }

    /// Add a variable binding to the context
    fn with_variable(mut self, name: Arc<str>, value: Value) -> Self {
        self.variables.insert(name, value);
        self
    }

    /// Get a variable by name
    fn get_variable(&self, name: &str) -> Option<&Value> {
        self.variables.get(name)
    }
}

/// What went wrong during evaluation.
///
/// This is the error of the resolver-based entry points ([`evaluate_with_resolver`],
/// [`evaluate_with_context`], [`evaluate_with_trace`], the `arena` evaluators) and of
/// the built-in functions. The higher-level entry points wrap it in [`HelError`], which
/// adds a line and column and a coarse [`ErrorKind`]; they convert via `From`, so `?` and
/// `.map_err(HelError::from)` work as you would expect.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum EvalError {
    /// An attribute was looked up and reported as unknown.
    ///
    /// The evaluators built into this crate never produce this: a resolver that answers
    /// `None` yields [`Value::Null`] instead, which fails the comparison rather than aborting
    /// evaluation. It exists for resolvers that want to report a genuinely invalid attribute
    /// path, and can be returned by a built-in.
    UnknownAttribute {
        /// Object half of the attribute path
        object: String,
        /// Field half of the attribute path
        field: String,
    },
    /// Type mismatch in operation
    TypeMismatch {
        /// Expected type
        expected: String,
        /// Actual type received
        got: String,
        /// Context where mismatch occurred
        context: String,
    },
    /// Invalid operation attempted
    InvalidOperation(String),
    /// Parse error occurred
    ParseError(String),
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::UnknownAttribute { object, field } => {
                write!(f, "Unknown attribute: {}.{}", object, field)
            }
            EvalError::TypeMismatch {
                expected,
                got,
                context,
            } => {
                write!(
                    f,
                    "Type mismatch in {}: expected {}, got {}",
                    context, expected, got
                )
            }
            EvalError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            EvalError::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for EvalError {}

/// Enhanced error type for HEL with line/column information
///
/// Returned by the high-level APIs - `validate_expression()`, `parse_expression()`,
/// `evaluate()`, `evaluate_script()` - with optional line and column numbers for parse errors.
///
/// # Examples
///
/// ```
/// use hel::validate_expression;
///
/// let bad_expr = "(unclosed";
/// match validate_expression(bad_expr) {
///     Err(e) => {
///         println!("Error: {}", e.message);
///         if let (Some(line), Some(col)) = (e.line, e.column) {
///             println!("At line {}, column {}", line, col);
///         }
///     }
///     Ok(_) => println!("Valid!"),
/// }
/// ```
#[derive(Debug, Clone)]
pub struct HelError {
    /// Error message describing what went wrong
    pub message: String,
    /// Line number where error occurred (for parse errors)
    pub line: Option<usize>,
    /// Column number where error occurred (for parse errors)
    pub column: Option<usize>,
    /// Kind of error that occurred
    pub kind: ErrorKind,
}

/// A coarse classification of a [`HelError`], for callers that branch on cause without
/// string-matching the message.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ErrorKind {
    /// The text was not a valid expression or script.
    ParseError,
    /// Evaluation reached a state it could not proceed from - most often a call to a
    /// function the context has no registry for.
    EvaluationError,
    /// An operand had the wrong type for the operator applied to it.
    TypeError,
    /// An attribute was reported as unknown; see
    /// [`EvalError::UnknownAttribute`] for why this is rare.
    UnknownAttribute,
}

impl HelError {
    /// Create a parse error without location information
    #[must_use]
    pub fn parse_error(message: String) -> Self {
        Self {
            message,
            line: None,
            column: None,
            kind: ErrorKind::ParseError,
        }
    }

    /// Create a parse error with line and column information
    #[must_use]
    pub fn parse_error_at(message: String, line: usize, column: usize) -> Self {
        Self {
            message,
            line: Some(line),
            column: Some(column),
            kind: ErrorKind::ParseError,
        }
    }

    /// Create an evaluation error
    #[must_use]
    pub fn eval_error(message: String) -> Self {
        Self {
            message,
            line: None,
            column: None,
            kind: ErrorKind::EvaluationError,
        }
    }

    /// Create a type error
    #[must_use]
    pub fn type_error(message: String) -> Self {
        Self {
            message,
            line: None,
            column: None,
            kind: ErrorKind::TypeError,
        }
    }

    /// Create an unknown attribute error
    #[must_use]
    pub fn unknown_attribute(message: String) -> Self {
        Self {
            message,
            line: None,
            column: None,
            kind: ErrorKind::UnknownAttribute,
        }
    }
}

impl std::fmt::Display for HelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let (Some(line), Some(column)) = (self.line, self.column) {
            write!(
                f,
                "HEL {:?} at line {}, column {}: {}",
                self.kind, line, column, self.message
            )
        } else {
            write!(f, "HEL {:?}: {}", self.kind, self.message)
        }
    }
}

impl std::error::Error for HelError {}

impl From<EvalError> for HelError {
    fn from(err: EvalError) -> Self {
        match err {
            EvalError::ParseError(msg) => HelError::parse_error(msg),
            EvalError::TypeMismatch {
                expected,
                got,
                context,
            } => HelError::type_error(format!(
                "Type mismatch in {}: expected {}, got {}",
                context, expected, got
            )),
            EvalError::UnknownAttribute { object, field } => {
                HelError::unknown_attribute(format!("Unknown attribute: {}.{}", object, field))
            }
            EvalError::InvalidOperation(msg) => HelError::eval_error(msg),
        }
    }
}

/// Parse a HEL expression into an AST (low-level API)
///
/// The whole of `input` must be a single valid HEL expression - trailing content is a
/// parse error, not something ignored. Prefer [`parse_expression`] or
/// [`validate_expression`], which report the failure as a `Result` instead of panicking.
///
/// # Panics
///
/// Panics if `input` is not a valid HEL expression.
///
/// # Examples
///
/// ```
/// use hel::parse_rule;
///
/// let ast = parse_rule(r#"binary.format == "elf""#);
/// ```
#[must_use]
pub fn parse_rule(input: &str) -> AstNode {
    let mut pairs = HelParser::parse(Rule::top, input).expect("parse error");
    build_ast(pairs.next().unwrap())
}

fn build_ast(pair: Pair<Rule>) -> AstNode {
    match pair.as_rule() {
        Rule::top | Rule::condition => {
            let mut inner = pair.into_inner();
            let next = inner.next().expect("Empty condition");
            build_ast(next)
        }

        Rule::logical_and | Rule::logical_or => {
            let is_and = pair.as_rule() == Rule::logical_and;
            let nodes: Vec<AstNode> = pair
                .into_inner()
                .filter_map(|inner| match inner.as_rule() {
                    Rule::and_op | Rule::or_op => None,
                    _ => Some(build_ast(inner)),
                })
                .collect();

            if is_and {
                AstNode::And(nodes)
            } else {
                AstNode::Or(nodes)
            }
        }

        Rule::comparison => {
            let mut inner = pair.into_inner();
            let left = build_ast(inner.next().expect("Missing left operand"));
            let op = parse_comparator(inner.next().expect("Missing comparator"));
            let right = build_ast(inner.next().expect("Missing right operand"));

            AstNode::Comparison {
                left: Box::new(left),
                op,
                right: Box::new(right),
            }
        }

        Rule::attribute_access => {
            let mut inner = pair.into_inner();
            let object = inner.next().expect("Missing object").as_str();
            let field = inner.next().expect("Missing field").as_str();
            AstNode::Attribute {
                object: object.into(),
                field: field.into(),
            }
        }

        Rule::literal => {
            let inner_pair = pair.into_inner().next().expect("Empty literal");
            build_ast(inner_pair)
        }

        Rule::string_literal => AstNode::String(pair.as_str().trim_matches('"').into()),

        Rule::float_literal => {
            let val = pair.as_str().parse::<f64>().expect("invalid float");
            AstNode::Float(val)
        }

        Rule::number_literal => {
            let num_str = pair.as_str();
            match parse_number(num_str) {
                Some(n) => AstNode::Number(n),
                None => panic!("Failed to parse number literal: '{}'", num_str),
            }
        }

        Rule::boolean_literal => AstNode::Bool(pair.as_str() == "true"),

        Rule::list_literal => {
            let elements: Vec<AstNode> = pair.into_inner().map(|p| build_ast(p)).collect();
            AstNode::ListLiteral(elements)
        }

        Rule::map_literal => {
            let mut entries = Vec::new();
            for entry_pair in pair.into_inner() {
                if entry_pair.as_rule() == Rule::map_entry {
                    let mut entry_inner = entry_pair.into_inner();
                    let key_pair = entry_inner.next().expect("Missing map key");
                    let key = key_pair.as_str().trim_matches('"').into();
                    let value = build_ast(entry_inner.next().expect("Missing map value"));
                    entries.push((key, value));
                }
            }
            AstNode::MapLiteral(entries)
        }

        Rule::function_call => {
            let mut inner = pair.into_inner();
            let first = inner.next().expect("Missing function name");

            // A second identifier before the argument list means the call is namespaced
            // (`ns.func(...)`); the grammar orders it first, so anything left after it is
            // an argument.
            let (namespace, name, remaining_args) = match inner.next() {
                Some(second) => (
                    Some(Arc::from(first.as_str())),
                    Arc::from(second.as_str()),
                    inner,
                ),
                None => (None, Arc::from(first.as_str()), inner),
            };

            let args: Vec<AstNode> = remaining_args.map(build_ast).collect();

            AstNode::FunctionCall {
                namespace,
                name,
                args,
            }
        }

        Rule::identifier | Rule::variable | Rule::symbolic => {
            AstNode::Identifier(pair.as_str().into())
        }

        Rule::primary | Rule::comparison_term | Rule::term | Rule::parenthesized => {
            build_ast(pair.into_inner().next().expect("Empty wrapper"))
        }

        _ => unreachable!("Unhandled rule: {:?}", pair.as_rule()),
    }
}

fn parse_comparator(pair: Pair<Rule>) -> Comparator {
    let token = pair.as_str().trim();
    match token {
        "==" => Comparator::Eq,
        "!=" => Comparator::Ne,
        ">" => Comparator::Gt,
        ">=" => Comparator::Ge,
        "<" => Comparator::Lt,
        "<=" => Comparator::Le,
        "CONTAINS" => Comparator::Contains,
        "IN" => Comparator::In,
        _ => panic!(
            "Unhandled comparator: {}. Supported comparators: ==, !=, >, >=, <, <=, CONTAINS, IN",
            token
        ),
    }
}

// ============================================================================
// Evaluation APIs (resolver-based, low-level)
// ============================================================================

/// Evaluate a HEL expression with a custom resolver (low-level API)
///
/// Evaluates `condition` against attribute values supplied by `resolver`. Built-in
/// functions are not available - use [`evaluate_with_context`] when the expression
/// calls any. Most callers are better served by [`evaluate`], which pairs with
/// [`FactsEvalContext`].
///
/// # Errors
///
/// Returns [`EvalError::ParseError`] if `condition` is not a valid HEL expression,
/// [`EvalError::InvalidOperation`] if it calls a function - this entry point holds no
/// registry, so *any* call is an error - and [`EvalError::TypeMismatch`] if an operand
/// has the wrong type for its operator or the expression as a whole is not a boolean.
///
/// An attribute the resolver answers `None` for is not an error: it resolves to
/// [`Value::Null`], which simply fails whatever comparison it appears in.
///
/// # Examples
///
/// ```
/// use hel::{evaluate_with_resolver, HelResolver, Value};
///
/// struct MyResolver;
/// impl HelResolver for MyResolver {
///     fn resolve_attr(&self, object: &str, field: &str) -> Option<Value> {
///         match (object, field) {
///             ("binary", "arch") => Some(Value::String("x86_64".into())),
///             _ => None,
///         }
///     }
/// }
///
/// let result = evaluate_with_resolver(
///     r#"binary.arch == "x86_64""#,
///     &MyResolver
/// ).expect("evaluation failed");
/// assert!(result);
/// ```
pub fn evaluate_with_resolver(
    condition: &str,
    resolver: &dyn HelResolver,
) -> Result<bool, EvalError> {
    validate_expression(condition).map_err(|e| EvalError::ParseError(e.to_string()))?;
    let ast = parse_rule(condition);
    let ctx = EvalContext::new(resolver);
    evaluate_ast_with_context(&ast, &ctx)
}

/// Evaluate a HEL expression with resolver and built-in functions (low-level API)
///
/// Like [`evaluate_with_resolver`], but the expression may also call the functions
/// held by `builtins`. Resolution of function names is namespace-qualified and
/// deterministic.
///
/// # Errors
///
/// Returns [`EvalError::ParseError`] if `condition` is not a valid HEL expression,
/// [`EvalError::InvalidOperation`] if the expression calls a function `builtins` does not
/// define or passes one arguments it rejects, and [`EvalError::TypeMismatch`] if an
/// operand has the wrong type for its operator or the expression as a whole is not a
/// boolean.
///
/// An attribute the resolver answers `None` for is not an error: it resolves to
/// [`Value::Null`], which simply fails whatever comparison it appears in.
///
/// # Examples
///
/// ```
/// use hel::{evaluate_with_context, HelResolver, Value, BuiltinsRegistry, CoreBuiltinsProvider};
///
/// struct MyResolver;
/// impl HelResolver for MyResolver {
///     fn resolve_attr(&self, _: &str, _: &str) -> Option<Value> { None }
/// }
///
/// let mut registry = BuiltinsRegistry::new();
/// registry.register(&CoreBuiltinsProvider).expect("register failed");
///
/// let result = evaluate_with_context(
///     r#"core.len(["a", "b"]) == 2"#,
///     &MyResolver,
///     &registry
/// ).expect("evaluation failed");
/// assert!(result);
/// ```
pub fn evaluate_with_context(
    condition: &str,
    resolver: &dyn HelResolver,
    builtins: &builtins::BuiltinsRegistry,
) -> Result<bool, EvalError> {
    validate_expression(condition).map_err(|e| EvalError::ParseError(e.to_string()))?;
    let ast = parse_rule(condition);
    let ctx = EvalContext::with_builtins(resolver, builtins);
    evaluate_ast_with_context(&ast, &ctx)
}

fn evaluate_ast_with_context(ast: &AstNode, ctx: &EvalContext) -> Result<bool, EvalError> {
    match ast {
        AstNode::Bool(b) => Ok(*b),
        AstNode::And(nodes) => {
            for node in nodes {
                if !evaluate_ast_with_context(node, ctx)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        AstNode::Or(nodes) => {
            for node in nodes {
                if evaluate_ast_with_context(node, ctx)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        AstNode::Comparison { left, op, right } => {
            evaluate_comparison_with_context(left, *op, right, ctx)
        }
        // Any other node is a value rather than a condition, so it is only usable as a
        // condition when that value is itself a boolean - otherwise the expression is a
        // type error rather than a silent false.
        other => {
            let value = eval_node_to_value_with_context(other, ctx)?;
            match value {
                Value::Bool(b) => Ok(b),
                _ => Err(EvalError::TypeMismatch {
                    expected: "boolean".to_string(),
                    got: format!("{:?}", value),
                    context: "boolean expression context".to_string(),
                }),
            }
        }
    }
}

fn evaluate_comparison_with_context(
    left: &AstNode,
    op: Comparator,
    right: &AstNode,
    ctx: &EvalContext,
) -> Result<bool, EvalError> {
    let left_val = eval_node_to_value_with_context(left, ctx)?;
    let right_val = eval_node_to_value_with_context(right, ctx)?;
    Ok(compare_new_values(&left_val, &right_val, op))
}

pub(crate) fn eval_node_to_value_with_context(
    node: &AstNode,
    ctx: &EvalContext,
) -> Result<Value, EvalError> {
    match node {
        AstNode::Bool(b) => Ok(Value::Bool(*b)),
        AstNode::String(s) => Ok(Value::String(s.clone())),
        AstNode::Number(n) => Ok(Value::Number(*n as f64)),
        AstNode::Float(f) => Ok(Value::Number(*f)),
        AstNode::Identifier(s) => {
            // A binding from a script's `let` shadows the bare word; an unbound identifier
            // reads as its own name, which is how unquoted words are written in a condition.
            if let Some(value) = ctx.get_variable(s) {
                Ok(value.clone())
            } else {
                Ok(Value::String(s.clone()))
            }
        }
        AstNode::Attribute { object, field } => Ok(ctx
            .resolver
            .resolve_attr(object, field)
            .unwrap_or(Value::Null)),
        AstNode::ListLiteral(elements) => {
            let values: Result<Vec<Value>, EvalError> = elements
                .iter()
                .map(|e| eval_node_to_value_with_context(e, ctx))
                .collect();
            Ok(Value::List(values?))
        }
        AstNode::MapLiteral(entries) => {
            let mut map = BTreeMap::new();
            for (key, value_node) in entries {
                let value = eval_node_to_value_with_context(value_node, ctx)?;
                map.insert(key.clone(), value);
            }
            Ok(Value::Map(map))
        }
        // A condition nested as an operand is re-wrapped as a Value so that, say,
        // `(a == 1) == true` has something to compare against.
        AstNode::Comparison { .. } | AstNode::And(_) | AstNode::Or(_) => {
            let bool_result = evaluate_ast_with_context(node, ctx)?;
            Ok(Value::Bool(bool_result))
        }
        AstNode::FunctionCall {
            namespace,
            name,
            args,
        } => {
            let arg_values: Result<Vec<Value>, EvalError> = args
                .iter()
                .map(|arg| eval_node_to_value_with_context(arg, ctx))
                .collect();
            let arg_values = arg_values?;

            if let Some(builtins) = ctx.builtins {
                let ns = namespace.as_ref().map(|s| s.as_ref()).unwrap_or("core");
                builtins.call(ns, name, &arg_values)
            } else {
                Err(EvalError::InvalidOperation(format!(
                    "Function calls not supported without built-ins registry: {}.{}",
                    namespace.as_ref().map(|s| s.as_ref()).unwrap_or("core"),
                    name
                )))
            }
        }
    }
}

pub(crate) fn compare_new_values(left: &Value, right: &Value, op: Comparator) -> bool {
    match op {
        Comparator::Eq => match (left, right) {
            (Value::Null, Value::Null) => true,
            (Value::Null, _) | (_, Value::Null) => false,
            (Value::Bool(l), Value::Bool(r)) => l == r,
            (Value::String(l), Value::String(r)) => l == r,
            (Value::Number(l), Value::Number(r)) => {
                if l.is_nan() || r.is_nan() {
                    return false;
                }
                l == r
            }
            _ => false,
        },
        Comparator::Ne => !compare_new_values(left, right, Comparator::Eq),
        Comparator::Contains => match (left, right) {
            (Value::String(l), Value::String(r)) => l.contains(&**r),
            (Value::List(list), val) => list
                .iter()
                .any(|item| compare_new_values(item, val, Comparator::Eq)),
            (Value::Map(map), Value::String(key)) => map.contains_key(key),
            _ => false,
        },
        Comparator::In => match (left, right) {
            (val, Value::List(list)) => list
                .iter()
                .any(|item| compare_new_values(val, item, Comparator::Eq)),
            (Value::String(s), Value::String(haystack)) => haystack.contains(&**s),
            _ => false,
        },
        Comparator::Gt | Comparator::Ge | Comparator::Lt | Comparator::Le => match (left, right) {
            (Value::Number(l), Value::Number(r)) => {
                if l.is_nan() || r.is_nan() {
                    return false;
                }
                match op {
                    Comparator::Gt => l > r,
                    Comparator::Ge => l >= r,
                    Comparator::Lt => l < r,
                    Comparator::Le => l <= r,
                    _ => false,
                }
            }
            _ => false,
        },
    }
}

fn parse_number(val: &str) -> Option<u64> {
    let val = val.trim();
    if let Some(stripped) = val.strip_prefix("0x").or_else(|| val.strip_prefix("0X")) {
        u64::from_str_radix(stripped, 16).ok()
    } else {
        val.parse::<u64>().ok()
    }
}

// ============================================================================
// Expression Validation, Parsing
// ============================================================================

/// Represents a parsed HEL expression
pub type Expression = AstNode;

/// Validates HEL expression syntax without evaluation
///
/// The whole of `expr` must be one expression: trailing characters are a parse error, not
/// something ignored. Use this to reject malformed rules before they are ever evaluated.
///
/// # Errors
///
/// Returns [`HelError`] carrying a line and column if `expr` is not a valid HEL
/// expression, whether the failure is bad syntax, an unterminated literal, or valid
/// syntax followed by extra input.
///
/// # Examples
///
/// ```
/// use hel::validate_expression;
///
/// let expr = r#"binary.arch == "x86_64" AND security.nx == false"#;
/// assert!(validate_expression(expr).is_ok());
///
/// // Trailing input is rejected rather than silently dropped.
/// assert!(validate_expression(r#"binary.arch == "x86_64" oops"#).is_err());
/// assert!(validate_expression("(").is_err());
/// ```
pub fn validate_expression(expr: &str) -> Result<(), HelError> {
    match HelParser::parse(Rule::top, expr) {
        Ok(_) => Ok(()),
        Err(e) => {
            let (line, column) = match &e.line_col {
                pest::error::LineColLocation::Pos((l, c)) => (*l, *c),
                pest::error::LineColLocation::Span((l, c), _) => (*l, *c),
            };

            Err(HelError::parse_error_at(
                format!("{}", e.variant),
                line,
                column,
            ))
        }
    }
}

/// Parse a HEL expression into an AST
///
/// Equivalent to [`validate_expression`] followed by [`parse_rule`], but the failure is a
/// `Result` rather than a panic.
///
/// The AST is for *inspection*: walking a rule, rewriting it, printing it, or checking what it
/// refers to. It does not make evaluation cheaper - every evaluator in this crate takes
/// expression text and parses it, so holding an `Expression` saves nothing on the evaluation
/// path. For repeated evaluation the lever is `hel::arena::ArenaParser`, which reuses the
/// memory the AST is built in.
///
/// # Errors
///
/// Returns [`HelError`] with line and column information if `expr` is not a valid HEL
/// expression. Because this validates before parsing, the returned error is never a panic.
///
/// # Examples
///
/// ```
/// use hel::parse_expression;
///
/// let expr = r#"binary.format == "elf""#;
/// let ast = parse_expression(expr).expect("parse failed");
///
/// assert!(parse_expression("binary.format ==").is_err());
/// ```
pub fn parse_expression(expr: &str) -> Result<Expression, HelError> {
    validate_expression(expr)?;
    Ok(parse_rule(expr))
}

/// The simplest [`HelResolver`]: a map of facts you fill in yourself.
///
/// # Key format
///
/// Keys are `"object.field"`, exactly as written in an expression. HEL's grammar only lets
/// you reference an attribute through `object.field`, so a key without a dot - `"arch"`
/// rather than `"binary.arch"` - can never be looked up. The lookup is an exact string
/// match: `"binary.arch"` and `"Binary.arch"` are different facts.
///
/// # Examples
///
/// ```
/// use hel::{evaluate, FactsEvalContext, Value};
///
/// let mut ctx = FactsEvalContext::new();
/// ctx.add_fact("binary.arch", Value::String("x86_64".into()));
/// ctx.add_fact("security.nx", Value::Bool(false));
///
/// assert!(evaluate(r#"binary.arch == "x86_64""#, &ctx).expect("evaluated"));
/// ```
pub struct FactsEvalContext {
    facts: BTreeMap<String, Value>,
}

impl FactsEvalContext {
    /// Create a context with no facts in it.
    #[must_use]
    pub fn new() -> Self {
        Self {
            facts: BTreeMap::new(),
        }
    }

    /// Set the fact at `key`, replacing any previous value.
    ///
    /// See the [type-level docs](FactsEvalContext#key-format) for the key format.
    pub fn add_fact(&mut self, key: &str, value: Value) {
        self.facts.insert(key.to_string(), value);
    }
}

impl Default for FactsEvalContext {
    fn default() -> Self {
        Self::new()
    }
}

impl HelResolver for FactsEvalContext {
    fn resolve_attr(&self, object: &str, field: &str) -> Option<Value> {
        let key = format!("{}.{}", object, field);
        self.facts.get(&key).cloned()
    }
}

/// Evaluate expression against context
///
/// The simplest entry point: parse `expr`, resolve its attributes from the facts in
/// `context`, and return the resulting boolean.
///
/// # Errors
///
/// Returns [`HelError`] if `expr` is not a valid HEL expression, if an operand has the
/// wrong type for its operator, or if it calls a function - this path has no built-ins, so
/// use [`evaluate_with_context`] for expressions that call them.
///
/// A fact that is not in `context` is not an error: it reads as [`Value::Null`].
///
/// # Examples
///
/// ```
/// use hel::{evaluate, FactsEvalContext, Value};
///
/// let mut ctx = FactsEvalContext::new();
/// ctx.add_fact("binary.arch", Value::String("x86_64".into()));
/// ctx.add_fact("security.nx", Value::Bool(false));
///
/// let expr = r#"binary.arch == "x86_64" AND security.nx == false"#;
/// let result = evaluate(expr, &ctx).expect("evaluation failed");
/// assert!(result);
/// ```
pub fn evaluate(expr: &str, context: &FactsEvalContext) -> Result<bool, HelError> {
    let ast = parse_expression(expr)?;
    let ctx = EvalContext::new(context);
    evaluate_ast_with_context(&ast, &ctx).map_err(|e| e.into())
}

// ============================================================================
// Script Support (Let Bindings and Multi-Expression Scripts)
// ============================================================================

/// Represents a parsed HEL script with let bindings
#[derive(Debug, Clone)]
pub struct Script {
    /// Let bindings in the script (name -> expression)
    pub bindings: Vec<(Arc<str>, AstNode)>,
    /// Final expression that must evaluate to a boolean
    pub final_expr: AstNode,
}

/// Parse and validate a .hel script file (may contain multiple expressions, let bindings)
///
/// A script is a sequence of `let <name> = <expression>` bindings followed by a final
/// boolean expression. Each binding is evaluated in order and may refer to the ones before
/// it. Blank lines and lines starting with `#` are ignored.
///
/// # Errors
///
/// Returns [`HelError`] if a binding's expression or the final expression is not valid HEL
/// syntax, if the script has no final expression, or if a line is neither a binding nor a
/// continuation and cannot be interpreted as one of those.
///
/// # Note
///
/// The parser is line-oriented, and a `let` binding absorbs the following lines only while
/// the next line begins with a joining operator (`AND`, `OR`, `&&`, `||`) or the text
/// collected so far is not yet a complete expression. So a binding keeps its continuation
/// only if the break is unambiguous - either the next line starts with an operator, or the
/// line before it ended mid-expression. Anything after the final expression is joined onto
/// it and therefore has to be a continuation of it.
///
/// # Examples
///
/// ```
/// use hel::parse_script;
///
/// let script = r#"
/// let has_perms = manifest.permissions CONTAINS "READ_SMS"
/// has_perms AND binary.entropy > 7.5
/// "#;
///
/// let parsed = parse_script(script).expect("parse failed");
/// ```
pub fn parse_script(script: &str) -> Result<Script, HelError> {
    let lines: Vec<&str> = script.lines().collect();
    let mut bindings = Vec::new();
    let mut final_expr = None;

    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();

        if line.is_empty() || line.starts_with('#') {
            i += 1;
            continue;
        }

        if let Some(rest) = line.strip_prefix("let ") {
            let rest = rest.trim();

            if let Some(eq_pos) = rest.find('=') {
                let name = rest[..eq_pos].trim();
                let expr_after_eq = rest[eq_pos + 1..].trim();
                let mut expr_str = String::new();

                if !expr_after_eq.is_empty() {
                    expr_str = expr_after_eq.to_string();
                }

                // A binding swallows following lines only while the break is unambiguous:
                // the next line starts with a joining operator, or what we hold so far is
                // not yet a complete expression. Otherwise this line is the final
                // expression and the binding ends here.
                i += 1;
                while i < lines.len() {
                    let next_line = lines[i].trim();

                    if next_line.is_empty() || next_line.starts_with('#') {
                        i += 1;
                        continue;
                    }

                    if next_line.starts_with("let ") {
                        break;
                    }

                    if !expr_str.is_empty()
                        && !next_line.starts_with("AND")
                        && !next_line.starts_with("OR")
                        && !next_line.starts_with("and")
                        && !next_line.starts_with("or")
                        && !next_line.starts_with("&&")
                        && !next_line.starts_with("||")
                        && parse_expression(&expr_str).is_ok()
                    {
                        break;
                    }

                    if !expr_str.is_empty() {
                        expr_str.push(' ');
                    }
                    expr_str.push_str(next_line);
                    i += 1;
                }

                let expr = parse_expression(&expr_str)?;
                bindings.push((Arc::from(name), expr));
                continue;
            }
        }

        if final_expr.is_none() {
            let mut expr_str = line.to_string();

            // Everything left belongs to the final expression; the loop below cannot run
            // twice because we break out of the outer loop once it is set.
            i += 1;
            while i < lines.len() {
                let next_line = lines[i].trim();
                if !next_line.is_empty() && !next_line.starts_with('#') {
                    if !expr_str.is_empty() {
                        expr_str.push(' ');
                    }
                    expr_str.push_str(next_line);
                }
                i += 1;
            }

            final_expr = Some(parse_expression(&expr_str)?);
            break;
        }

        i += 1;
    }

    let final_expr = final_expr.ok_or_else(|| {
        HelError::parse_error("Script must have a final boolean expression".to_string())
    })?;

    Ok(Script {
        bindings,
        final_expr,
    })
}

/// Evaluate a script and return the final boolean result
///
/// Evaluates each `let` binding in declaration order, then evaluates the final expression.
/// A binding sees the facts in `context` and the bindings declared before it.
///
/// # Errors
///
/// Returns [`HelError`] if the script does not parse (see [`parse_script`]), or if
/// evaluation fails: an operand has the wrong type for its operator, or the script calls a
/// function - this path has no built-ins, so use [`evaluate_with_context`] with a
/// [`BuiltinsRegistry`] when a script needs them.
///
/// A fact absent from `context` is not an error: it reads as [`Value::Null`].
///
/// # Examples
///
/// ```
/// use hel::{evaluate_script, FactsEvalContext, Value};
///
/// let mut ctx = FactsEvalContext::new();
/// ctx.add_fact("manifest.permissions", Value::List(vec![
///     Value::String("READ_SMS".into()),
///     Value::String("SEND_SMS".into()),
/// ]));
/// ctx.add_fact("binary.entropy", Value::Number(8.0));
///
/// let script = r#"
/// let has_sms_perms = manifest.permissions CONTAINS "READ_SMS"
/// has_sms_perms AND binary.entropy > 7.5
/// "#;
///
/// let result = evaluate_script(script, &ctx).expect("evaluation failed");
/// assert!(result);
/// ```
pub fn evaluate_script(script: &str, context: &FactsEvalContext) -> Result<bool, HelError> {
    let parsed = parse_script(script)?;

    // Bindings are threaded through in declaration order, each one seeing those before it.
    let mut eval_ctx = EvalContext::new(context);
    for (name, expr) in &parsed.bindings {
        let value = eval_node_to_value_with_context(expr, &eval_ctx).map_err(HelError::from)?;
        eval_ctx = eval_ctx.with_variable(name.clone(), value);
    }

    evaluate_ast_with_context(&parsed.final_expr, &eval_ctx).map_err(|e| e.into())
}

// ============================================================================
// Convenience conversions into `Value`
// ============================================================================

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::String(Arc::from(s))
    }
}

impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::String(Arc::from(s.as_str()))
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}

impl From<f64> for Value {
    fn from(n: f64) -> Self {
        Value::Number(n)
    }
}

impl From<i32> for Value {
    fn from(n: i32) -> Self {
        Value::Number(f64::from(n))
    }
}

/// Widening an integer to `f64` is lossy above 2^53, which is the precision limit of the
/// mantissa. Facts and literals beyond that are not represented exactly.
impl From<u64> for Value {
    fn from(n: u64) -> Self {
        Value::Number(n as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `evaluate_with_trace` is covered in `crate::trace`; these cover this module's surface.

    #[test]
    fn test_resolver_number_and_list_behavior() {
        struct CustomResolver;
        impl HelResolver for CustomResolver {
            fn resolve_attr(&self, object: &str, field: &str) -> Option<Value> {
                if object == "enrichment" && field == "confidence" {
                    Some(Value::Number(0.85))
                } else if object == "tags" && field == "values" {
                    Some(Value::List(vec![
                        Value::String("security".into()),
                        Value::String("critical".into()),
                    ]))
                } else {
                    None
                }
            }
        }

        let resolver = CustomResolver;
        let cond1 = "enrichment.confidence > 0.7";
        let res1 = evaluate_with_resolver(cond1, &resolver).expect("evaluation failed");
        assert!(res1);

        let cond2 = r#"tags.values CONTAINS "critical""#;
        let res2 = evaluate_with_resolver(cond2, &resolver).expect("evaluation failed");
        assert!(res2);
    }

    #[test]
    fn test_nan_comparison_behavior() {
        struct NaNResolver;
        impl HelResolver for NaNResolver {
            fn resolve_attr(&self, object: &str, field: &str) -> Option<Value> {
                if object == "test" && field == "nan" {
                    Some(Value::Number(f64::NAN))
                } else {
                    None
                }
            }
        }

        let resolver = NaNResolver;
        let cond = "test.nan > 0.0";
        let res = evaluate_with_resolver(cond, &resolver).expect("evaluation failed");
        assert!(!res, "NaN comparison should be false");
    }

    // ========================================================================
    // Tests for new API functions
    // ========================================================================

    #[test]
    fn test_validate_expression_success() {
        let expr = r#"binary.arch == "x86_64" AND security.nx == false"#;
        assert!(validate_expression(expr).is_ok());
    }

    #[test]
    fn test_validate_expression_failure() {
        let bad_expr = "(";
        let result = validate_expression(bad_expr);
        assert!(result.is_err());

        if let Err(e) = result {
            assert!(e.line.is_some());
            assert!(e.column.is_some());
        }
    }

    /// The grammar is anchored with `SOI`/`EOI`, so an input must be *entirely* one valid
    /// expression. Without the anchors pest matches a prefix: `x == 5 garbage` would validate
    /// as the shorter expression `x`, with the trailing text silently dropped.
    #[test]
    fn test_validate_expression_rejects_partial_matches() {
        for bad in [
            "x ==",
            "x =",
            "x == 5 AND",
            "x == 5 garbage",
            "AND x == 5",
            "x == 5 == 6",
            "hello world",
            "core.len(]",
            "x == 5)",
            "(x == 5",
            "",
            "   ",
        ] {
            assert!(
                validate_expression(bad).is_err(),
                "{:?} should not validate",
                bad
            );
        }
    }

    #[test]
    fn test_validate_expression_accepts_full_expressions() {
        for good in [
            r#"binary.arch == "x86_64""#,
            "  x == 5  ",
            "x == 5\n",
            r#"(a == 1 AND b == 2) OR c == 3"#,
            r#"core.len(["a","b"]) == 2"#,
            r#"manifest.permissions CONTAINS "A""#,
            "$v == 1",
            "%s == 1",
        ] {
            assert!(
                validate_expression(good).is_ok(),
                "{:?} should validate",
                good
            );
        }
    }

    /// Malformed input reaches the low-level evaluators as `Err`, not a panic. `parse_rule` is
    /// the exception.
    #[test]
    fn test_evaluators_report_parse_errors_instead_of_panicking() {
        struct EmptyResolver;
        impl HelResolver for EmptyResolver {
            fn resolve_attr(&self, _: &str, _: &str) -> Option<Value> {
                None
            }
        }

        let result = evaluate_with_resolver("x == 5 garbage", &EmptyResolver);
        assert!(
            matches!(result, Err(EvalError::ParseError(_))),
            "got {:?}",
            result
        );

        let mut registry = builtins::BuiltinsRegistry::new();
        registry
            .register(&builtins::CoreBuiltinsProvider)
            .expect("register failed");
        let result = evaluate_with_context("x ==", &EmptyResolver, &registry);
        assert!(
            matches!(result, Err(EvalError::ParseError(_))),
            "got {:?}",
            result
        );

        let result = trace::evaluate_with_trace("x == 5 AND", &EmptyResolver, None);
        assert!(
            matches!(result, Err(EvalError::ParseError(_))),
            "got {:?}",
            result
        );
    }

    #[test]
    fn test_parse_expression_success() {
        let expr = r#"binary.format == "elf""#;
        let ast = parse_expression(expr).expect("parse failed");

        // The top node may be wrapped in `Or`/`And`; what this asserts is that it parsed.
        if let AstNode::Comparison { op, .. } = &ast {
            assert_eq!(*op, Comparator::Eq);
        }
    }

    #[test]
    fn test_facts_eval_context() {
        let mut ctx = FactsEvalContext::new();
        ctx.add_fact("binary.arch", Value::String("x86_64".into()));
        ctx.add_fact("security.nx", Value::Bool(false));

        assert_eq!(
            ctx.resolve_attr("binary", "arch"),
            Some(Value::String("x86_64".into()))
        );
        assert_eq!(ctx.resolve_attr("security", "nx"), Some(Value::Bool(false)));
    }

    #[test]
    fn test_evaluate_with_facts_context() {
        let mut ctx = FactsEvalContext::new();
        ctx.add_fact("binary.arch", "x86_64".into());
        ctx.add_fact("security.nx", false.into());

        let expr = r#"binary.arch == "x86_64" AND security.nx == false"#;
        let result = evaluate(expr, &ctx).expect("evaluation failed");
        assert!(result);
    }

    #[test]
    fn test_evaluate_with_facts_context_false() {
        let mut ctx = FactsEvalContext::new();
        ctx.add_fact("binary.arch", "arm".into());
        ctx.add_fact("security.nx", true.into());

        let expr = r#"binary.arch == "x86_64" AND security.nx == false"#;
        let result = evaluate(expr, &ctx).expect("evaluation failed");
        assert!(!result);
    }

    #[test]
    fn test_parse_script_simple() {
        let script = r#"
            let has_perms = manifest.permissions CONTAINS "READ_SMS"
            has_perms AND binary.entropy > 7.5
        "#;

        let parsed = parse_script(script).expect("parse failed");
        assert_eq!(parsed.bindings.len(), 1);
        assert_eq!(parsed.bindings[0].0.as_ref(), "has_perms");
    }

    #[test]
    fn test_parse_script_with_comments() {
        let script = r#"
            # This is a comment
            let has_perms = manifest.permissions CONTAINS "READ_SMS"

            # Another comment
            has_perms AND binary.entropy > 7.5
        "#;

        let parsed = parse_script(script).expect("parse failed");
        assert_eq!(parsed.bindings.len(), 1);
    }

    #[test]
    fn test_parse_script_multiple_bindings() {
        let script = r#"
            let has_sms_perms = manifest.permissions CONTAINS "READ_SMS"
            let has_obfuscation = binary.entropy > 7.5
            has_sms_perms AND has_obfuscation
        "#;

        let parsed = parse_script(script).expect("parse failed");
        assert_eq!(parsed.bindings.len(), 2);
        assert_eq!(parsed.bindings[0].0.as_ref(), "has_sms_perms");
        assert_eq!(parsed.bindings[1].0.as_ref(), "has_obfuscation");
    }

    #[test]
    fn test_evaluate_script_simple() {
        let mut ctx = FactsEvalContext::new();
        ctx.add_fact(
            "manifest.permissions",
            Value::List(vec![
                Value::String("READ_SMS".into()),
                Value::String("SEND_SMS".into()),
            ]),
        );
        ctx.add_fact("binary.entropy", Value::Number(8.0));

        let script = r#"
            let has_sms_perms = manifest.permissions CONTAINS "READ_SMS"
            has_sms_perms AND binary.entropy > 7.5
        "#;

        let result = evaluate_script(script, &ctx).expect("evaluation failed");
        assert!(result);
    }

    #[test]
    fn test_evaluate_script_with_multiple_bindings() {
        let mut ctx = FactsEvalContext::new();
        ctx.add_fact(
            "manifest.permissions",
            Value::List(vec![
                Value::String("READ_SMS".into()),
                Value::String("SEND_SMS".into()),
            ]),
        );
        ctx.add_fact("binary.entropy", Value::Number(8.0));
        ctx.add_fact("strings.count", Value::Number(5.0));

        let script = r#"
            let has_sms_perms = manifest.permissions CONTAINS "READ_SMS" AND manifest.permissions CONTAINS "SEND_SMS"
            let has_obfuscation = binary.entropy > 7.5 OR strings.count < 10
            has_sms_perms AND has_obfuscation
        "#;

        let result = evaluate_script(script, &ctx).expect("evaluation failed");
        assert!(result);
    }

    #[test]
    fn test_value_from_conversions() {
        let v1: Value = "test".into();
        assert_eq!(v1, Value::String("test".into()));

        let v2: Value = true.into();
        assert_eq!(v2, Value::Bool(true));

        let v3: Value = 42.5.into();
        assert_eq!(v3, Value::Number(42.5));

        let v4: Value = 42i32.into();
        assert_eq!(v4, Value::Number(42.0));
    }

    #[test]
    fn test_eval_context_variables() {
        let ctx = FactsEvalContext::new();
        let mut eval_ctx = EvalContext::new(&ctx);

        eval_ctx = eval_ctx.with_variable(Arc::from("test_var"), Value::Bool(true));

        let result = eval_ctx.get_variable("test_var");
        assert_eq!(result, Some(&Value::Bool(true)));
    }

    #[test]
    fn test_script_let_binding_storage() {
        let ctx = FactsEvalContext::new();
        let mut eval_ctx = EvalContext::new(&ctx);

        let name: Arc<str> = Arc::from("has_perms");
        let value = Value::Bool(true);

        eval_ctx = eval_ctx.with_variable(name.clone(), value);

        let retrieved = eval_ctx.get_variable("has_perms");
        assert_eq!(retrieved, Some(&Value::Bool(true)));

        let identifier = AstNode::Identifier(Arc::from("has_perms"));
        let result = eval_node_to_value_with_context(&identifier, &eval_ctx).unwrap();
        assert_eq!(result, Value::Bool(true));
    }
}
