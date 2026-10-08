//! Arena-allocated AST for HEL.
//!
//! Nodes are bump-allocated into one arena and referenced as `&'arena`, so a parse costs a
//! pointer bump instead of one heap allocation per node, the nodes sit adjacent in memory, and
//! [`reset`](ArenaParser::reset) frees the lot in one step. This pays off when many expressions
//! go through one reused parser; a single one-off parse does not need it.
//!
//! # Example
//!
//! ```
//! use hel::arena::{ArenaParser, evaluate_arena};
//! use hel::{FactsEvalContext, Value};
//!
//! let mut ctx = FactsEvalContext::new();
//! ctx.add_fact("vars.x", Value::Number(10.0));
//! ctx.add_fact("vars.y", Value::Number(20.0));
//!
//! let parser = ArenaParser::new();
//! let result = evaluate_arena(r#"vars.x < vars.y"#, &ctx, &parser).expect("evaluation failed");
//! assert!(result);
//! ```
//!
//! # Arena reuse
//!
//! Reuse one `ArenaParser` across expressions, calling `reset()` between them:
//!
//! ```
//! use hel::arena::{ArenaParser, evaluate_arena};
//! use hel::{FactsEvalContext, Value};
//!
//! let mut ctx = FactsEvalContext::new();
//! ctx.add_fact("vars.x", Value::Number(10.0));
//!
//! let mut parser = ArenaParser::new();
//!
//! let result1 = evaluate_arena(r#"vars.x == 10"#, &ctx, &parser).expect("eval failed");
//! assert!(result1);
//!
//! parser.reset();
//!
//! let result2 = evaluate_arena(r#"vars.x > 5"#, &ctx, &parser).expect("eval failed");
//! assert!(result2);
//! ```

use bumpalo::collections::Vec as BumpVec;
use bumpalo::Bump;

use crate::{
    builtins::BuiltinsRegistry, Comparator, EvalError, FactsEvalContext, HelError, HelParser,
    HelResolver, Rule, Value,
};
use pest::Parser;

// ============================================================================
// Arena AST Types
// ============================================================================

/// An arena-allocated AST node - the counterpart of [`hel::AstNode`](crate::AstNode), with
/// children as arena references (`&'arena`) rather than `Box` / `Vec` / `Arc`. The `'arena`
/// lifetime ties every node to its arena, so a node cannot outlive the memory it points into.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum AstNode<'arena> {
    /// Boolean literal (true or false)
    Bool(bool),
    /// String literal (arena-allocated str slice)
    String(&'arena str),
    /// Integer number literal
    Number(u64),
    /// Float number (f64)
    Float(f64),
    /// Identifier (variable name or unqualified reference)
    Identifier(&'arena str),
    /// Attribute access (object.field notation)
    Attribute {
        /// Object name
        object: &'arena str,
        /// Field name
        field: &'arena str,
    },
    /// Comparison expression (left op right)
    Comparison {
        /// Left operand (arena ref instead of Box)
        left: &'arena AstNode<'arena>,
        /// Comparison operator
        op: Comparator,
        /// Right operand (arena ref instead of Box)
        right: &'arena AstNode<'arena>,
    },
    /// Logical AND expression (arena slice instead of Vec)
    And(&'arena [AstNode<'arena>]),
    /// Logical OR expression (arena slice instead of Vec)
    Or(&'arena [AstNode<'arena>]),
    /// List literal: [1, 2, 3] or ["a", "b"]
    ListLiteral(&'arena [AstNode<'arena>]),
    /// Map literal: {"key": value, ...}
    MapLiteral(&'arena [(&'arena str, AstNode<'arena>)]),
    /// Function call: namespace.function(args) or function(args)
    FunctionCall {
        /// Namespace (if qualified, e.g., "core" in core.len)
        namespace: Option<&'arena str>,
        /// Function name
        name: &'arena str,
        /// Arguments (arena slice instead of Vec)
        args: &'arena [AstNode<'arena>],
    },
}

// ============================================================================
// Arena Parser
// ============================================================================

/// Parser that builds arena-allocated ASTs into the arena it owns.
///
/// # Example
///
/// ```
/// use hel::arena::ArenaParser;
///
/// let parser = ArenaParser::new();
/// let ast = parser.parse_rule(r#"x == 10 AND y > 5"#);
/// // ast is a reference into the parser's arena
/// ```
pub struct ArenaParser {
    arena: Bump,
}

impl ArenaParser {
    /// Create a new arena parser
    #[must_use]
    pub fn new() -> Self {
        Self { arena: Bump::new() }
    }

    /// Parse a HEL rule into an arena-allocated AST
    ///
    /// The returned node borrows from `self` and stays valid until the parser is
    /// reset or dropped. Prefer [`ArenaParser::parse_expression`] unless the input is
    /// already known to be well-formed.
    ///
    /// # Panics
    ///
    /// Panics if `input` fails to parse.
    ///
    /// # Examples
    ///
    /// ```
    /// use hel::arena::ArenaParser;
    ///
    /// let parser = ArenaParser::new();
    /// let ast = parser.parse_rule(r#"x == 10 AND y > 5"#);
    /// // `ast` borrows from the parser's arena rather than the heap.
    /// assert!(format!("{ast:?}").contains("And"));
    /// ```
    pub fn parse_rule<'a>(&'a self, input: &str) -> &'a AstNode<'a> {
        let mut pairs = HelParser::parse(Rule::top, input)
            .unwrap_or_else(|e| panic!("Failed to parse expression: {}", e));
        self.build_ast_arena(pairs.next().unwrap())
    }

    /// Parse a HEL expression with validation
    ///
    /// The fallible counterpart to [`ArenaParser::parse_rule`]: the input is validated
    /// before the AST is built, so malformed expressions yield `Err` instead of a panic.
    ///
    /// # Errors
    ///
    /// Returns [`HelError`] with line and column information if `expr` is not a valid
    /// HEL expression.
    ///
    /// # Examples
    ///
    /// ```
    /// use hel::arena::ArenaParser;
    ///
    /// let parser = ArenaParser::new();
    /// assert!(parser.parse_expression(r#"x == 10"#).is_ok());
    /// assert!(parser.parse_expression("x ==").is_err());
    /// ```
    pub fn parse_expression<'a>(&'a self, expr: &str) -> Result<&'a AstNode<'a>, HelError> {
        crate::validate_expression(expr)?;
        Ok(self.parse_rule(expr))
    }

    /// Reset the arena for reuse.
    ///
    /// Frees every allocation in one step and makes the memory available to later parses,
    /// which is cheaper than dropping the parser and building a new one.
    ///
    /// # Warning
    ///
    /// Every AST node previously returned by this parser is invalidated. The borrow checker
    /// enforces this - `reset` takes `&mut self` while a live node holds `&self` - so safe code
    /// cannot carry a node across this call.
    pub fn reset(&mut self) {
        self.arena.reset();
    }

    /// Build an arena-allocated AST from a pest Pair
    fn build_ast_arena<'a>(&'a self, pair: pest::iterators::Pair<Rule>) -> &'a AstNode<'a> {
        let node = match pair.as_rule() {
            Rule::top | Rule::condition => {
                let mut inner = pair.into_inner();
                let next = inner.next().expect("Empty condition");
                return self.build_ast_arena(next);
            }

            Rule::logical_and | Rule::logical_or => {
                let is_and = pair.as_rule() == Rule::logical_and;
                let mut nodes = BumpVec::new_in(&self.arena);

                for inner in pair.into_inner() {
                    match inner.as_rule() {
                        Rule::and_op | Rule::or_op => {}
                        _ => nodes.push(*self.build_ast_arena(inner)),
                    }
                }

                let slice = nodes.into_bump_slice();
                if is_and {
                    AstNode::And(slice)
                } else {
                    AstNode::Or(slice)
                }
            }

            Rule::comparison => {
                let mut inner = pair.into_inner();
                let left = self.build_ast_arena(inner.next().expect("Missing left operand"));
                let op = parse_comparator(inner.next().expect("Missing comparator"));
                let right = self.build_ast_arena(inner.next().expect("Missing right operand"));

                AstNode::Comparison { left, op, right }
            }

            Rule::attribute_access => {
                let mut inner = pair.into_inner();
                let object = inner.next().expect("Missing object").as_str();
                let field = inner.next().expect("Missing field").as_str();
                AstNode::Attribute {
                    object: self.arena.alloc_str(object),
                    field: self.arena.alloc_str(field),
                }
            }

            Rule::literal => {
                let inner_pair = pair.into_inner().next().expect("Empty literal");
                return self.build_ast_arena(inner_pair);
            }

            Rule::string_literal => {
                let s = pair.as_str().trim_matches('"');
                AstNode::String(self.arena.alloc_str(s))
            }

            Rule::float_literal => {
                let val = pair.as_str().parse::<f64>().expect("invalid float");
                AstNode::Float(val)
            }

            Rule::number_literal => {
                let num_str = pair.as_str();
                match parse_number(num_str) {
                    Some(n) => AstNode::Number(n),
                    None => panic!(
                        "Failed to parse number literal: '{}'. Expected decimal or hexadecimal (0x prefix) integer",
                        num_str
                    ),
                }
            }

            Rule::boolean_literal => AstNode::Bool(pair.as_str() == "true"),

            Rule::list_literal => {
                let mut elements = BumpVec::new_in(&self.arena);
                for p in pair.into_inner() {
                    elements.push(*self.build_ast_arena(p));
                }
                AstNode::ListLiteral(elements.into_bump_slice())
            }

            Rule::map_literal => {
                let mut entries = BumpVec::new_in(&self.arena);
                for entry_pair in pair.into_inner() {
                    if entry_pair.as_rule() == Rule::map_entry {
                        let mut entry_inner = entry_pair.into_inner();
                        let key_pair = entry_inner.next().expect("Missing map key");
                        let key: &str = self.arena.alloc_str(key_pair.as_str().trim_matches('"'));
                        let value =
                            *self.build_ast_arena(entry_inner.next().expect("Missing map value"));
                        entries.push((key, value));
                    }
                }
                AstNode::MapLiteral(entries.into_bump_slice())
            }

            Rule::function_call => {
                let mut inner = pair.into_inner();
                let first = inner.next().expect("Missing function name");

                // A second identifier before the argument list means the call is namespaced
                // (`ns.func(...)`); the grammar orders it first, so anything left after it is
                // an argument.
                let (namespace, name, remaining_args): (Option<&str>, &str, _) = match inner.next()
                {
                    Some(second) => (
                        Some(self.arena.alloc_str(first.as_str())),
                        self.arena.alloc_str(second.as_str()),
                        inner,
                    ),
                    None => (None, self.arena.alloc_str(first.as_str()), inner),
                };

                let mut args = BumpVec::new_in(&self.arena);
                for arg in remaining_args {
                    args.push(*self.build_ast_arena(arg));
                }

                AstNode::FunctionCall {
                    namespace,
                    name,
                    args: args.into_bump_slice(),
                }
            }

            Rule::identifier | Rule::variable | Rule::symbolic => {
                AstNode::Identifier(self.arena.alloc_str(pair.as_str()))
            }

            Rule::primary | Rule::comparison_term | Rule::term | Rule::parenthesized => {
                return self.build_ast_arena(pair.into_inner().next().expect("Empty wrapper"));
            }

            _ => unreachable!("Unhandled rule: {:?}", pair.as_rule()),
        };

        self.arena.alloc(node)
    }
}

impl Default for ArenaParser {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn parse_comparator(pair: pest::iterators::Pair<Rule>) -> Comparator {
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

fn parse_number(val: &str) -> Option<u64> {
    let val = val.trim();
    if let Some(stripped) = val.strip_prefix("0x").or_else(|| val.strip_prefix("0X")) {
        u64::from_str_radix(stripped, 16).ok()
    } else {
        val.parse::<u64>().ok()
    }
}

// ============================================================================
// Arena Evaluation Context
// ============================================================================

/// Evaluation context for arena-based AST evaluation
struct ArenaEvalContext<'a> {
    resolver: &'a dyn HelResolver,
    builtins: Option<&'a BuiltinsRegistry>,
}

impl<'a> ArenaEvalContext<'a> {
    fn new(resolver: &'a dyn HelResolver) -> Self {
        Self {
            resolver,
            builtins: None,
        }
    }

    fn with_builtins(resolver: &'a dyn HelResolver, builtins: &'a BuiltinsRegistry) -> Self {
        Self {
            resolver,
            builtins: Some(builtins),
        }
    }
}

// ============================================================================
// Arena Evaluation Functions
// ============================================================================

/// Evaluate a HEL expression using arena-allocated AST
///
/// High-level convenience function: parses `expr` into `parser`'s arena and evaluates it
/// against the facts in `context`. Because the arena owns the AST, `parser` can be reused
/// for the next expression after a call to [`ArenaParser::reset`].
///
/// # Errors
///
/// Returns [`HelError`] if `expr` is not a valid HEL expression, if an operand has the
/// wrong type for its operator, or if the expression calls a function - built-ins are not
/// available on this path. An attribute absent from `context` is *not* an error; it reads
/// as [`Value::Null`].
///
/// # Example
///
/// ```
/// use hel::arena::{ArenaParser, evaluate_arena};
/// use hel::{FactsEvalContext, Value};
///
/// let mut ctx = FactsEvalContext::new();
/// ctx.add_fact("data.x", Value::Number(42.0));
///
/// let parser = ArenaParser::new();
/// let result = evaluate_arena(r#"data.x == 42"#, &ctx, &parser).expect("eval failed");
/// assert!(result);
/// ```
pub fn evaluate_arena(
    expr: &str,
    context: &FactsEvalContext,
    parser: &ArenaParser,
) -> Result<bool, HelError> {
    let ast = parser.parse_expression(expr)?;
    let ctx = ArenaEvalContext::new(context);
    evaluate_ast_arena(ast, &ctx).map_err(|e| e.into())
}

/// Evaluate arena AST with a resolver
///
/// Low-level API for evaluating an arena-allocated AST with a custom resolver.
/// Built-in functions are not available - use [`evaluate_with_context_arena`] when
/// the expression calls any.
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
pub fn evaluate_with_resolver_arena(
    condition: &str,
    resolver: &dyn HelResolver,
    parser: &ArenaParser,
) -> Result<bool, EvalError> {
    let ast = parser
        .parse_expression(condition)
        .map_err(|e| EvalError::ParseError(e.to_string()))?;
    let ctx = ArenaEvalContext::new(resolver);
    evaluate_ast_arena(ast, &ctx)
}

/// Evaluate arena AST with resolver and builtins
///
/// Low-level API for evaluating an arena-allocated AST with a custom resolver
/// and built-in functions.
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
pub fn evaluate_with_context_arena(
    condition: &str,
    resolver: &dyn HelResolver,
    builtins: &BuiltinsRegistry,
    parser: &ArenaParser,
) -> Result<bool, EvalError> {
    let ast = parser
        .parse_expression(condition)
        .map_err(|e| EvalError::ParseError(e.to_string()))?;
    let ctx = ArenaEvalContext::with_builtins(resolver, builtins);
    evaluate_ast_arena(ast, &ctx)
}

fn evaluate_ast_arena<'arena>(
    ast: &AstNode<'arena>,
    ctx: &ArenaEvalContext,
) -> Result<bool, EvalError> {
    match ast {
        AstNode::Bool(b) => Ok(*b),
        AstNode::And(nodes) => {
            for node in *nodes {
                if !evaluate_ast_arena(node, ctx)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        AstNode::Or(nodes) => {
            for node in *nodes {
                if evaluate_ast_arena(node, ctx)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        AstNode::Comparison { left, op, right } => evaluate_comparison_arena(left, *op, right, ctx),
        other => {
            let value = eval_node_to_value_arena(other, ctx)?;
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

fn evaluate_comparison_arena<'arena>(
    left: &AstNode<'arena>,
    op: Comparator,
    right: &AstNode<'arena>,
    ctx: &ArenaEvalContext,
) -> Result<bool, EvalError> {
    let left_val = eval_node_to_value_arena(left, ctx)?;
    let right_val = eval_node_to_value_arena(right, ctx)?;
    Ok(crate::compare_new_values(&left_val, &right_val, op))
}

fn eval_node_to_value_arena<'arena>(
    node: &AstNode<'arena>,
    ctx: &ArenaEvalContext,
) -> Result<Value, EvalError> {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    match node {
        AstNode::Bool(b) => Ok(Value::Bool(*b)),
        AstNode::String(s) => Ok(Value::String(Arc::from(*s))),
        AstNode::Number(n) => Ok(Value::Number(*n as f64)),
        AstNode::Float(f) => Ok(Value::Number(*f)),
        AstNode::Identifier(s) => {
            // This evaluator has no variable bindings, so a bare identifier can only be a
            // string literal. The heap evaluator consults its bindings first and falls back
            // to the same interpretation, so an identifier only differs between the two
            // when the expression is a script binding - which this path does not support.
            Ok(Value::String(Arc::from(*s)))
        }
        AstNode::Attribute { object, field } => Ok(ctx
            .resolver
            .resolve_attr(object, field)
            .unwrap_or(Value::Null)),
        AstNode::ListLiteral(elements) => {
            let values: Result<Vec<Value>, EvalError> = elements
                .iter()
                .map(|e| eval_node_to_value_arena(e, ctx))
                .collect();
            Ok(Value::List(values?))
        }
        AstNode::MapLiteral(entries) => {
            let mut map = BTreeMap::new();
            for (key, value_node) in *entries {
                let value = eval_node_to_value_arena(value_node, ctx)?;
                map.insert(Arc::from(*key), value);
            }
            Ok(Value::Map(map))
        }
        // A condition nested as an operand is re-wrapped as a Value so that, say,
        // `(a == 1) == true` has something to compare against.
        AstNode::Comparison { .. } | AstNode::And(_) | AstNode::Or(_) => {
            let bool_result = evaluate_ast_arena(node, ctx)?;
            Ok(Value::Bool(bool_result))
        }
        AstNode::FunctionCall {
            namespace,
            name,
            args,
        } => {
            let arg_values: Result<Vec<Value>, EvalError> = args
                .iter()
                .map(|arg| eval_node_to_value_arena(arg, ctx))
                .collect();
            let arg_values = arg_values?;

            if let Some(builtins) = ctx.builtins {
                let ns = namespace.unwrap_or("core");
                builtins.call(ns, name, &arg_values)
            } else {
                Err(EvalError::InvalidOperation(format!(
                    "Function calls not supported without built-ins registry: {}.{}",
                    namespace.unwrap_or("core"),
                    name
                )))
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FactsEvalContext, Value};
    use std::sync::Arc;

    #[test]
    fn test_arena_parse_simple_boolean() {
        let parser = ArenaParser::new();
        let ast = parser.parse_rule("true");

        // The grammar always wraps in Or([And([...])])
        match ast {
            AstNode::Or(ors) => {
                assert_eq!(ors.len(), 1);
                match &ors[0] {
                    AstNode::And(ands) => {
                        assert_eq!(ands.len(), 1);
                        assert!(matches!(ands[0], AstNode::Bool(true)));
                    }
                    _ => panic!("Expected And, got {:?}", ors[0]),
                }
            }
            _ => panic!("Expected Or, got {:?}", ast),
        }
    }

    #[test]
    fn test_arena_parse_comparison() {
        let parser = ArenaParser::new();
        let ast = parser.parse_rule(r#"x == 10"#);

        // The grammar always wraps in Or([And([...])])
        match ast {
            AstNode::Or(ors) => {
                assert_eq!(ors.len(), 1);
                match &ors[0] {
                    AstNode::And(ands) => {
                        assert_eq!(ands.len(), 1);
                        match &ands[0] {
                            AstNode::Comparison { left, op, right } => {
                                assert!(matches!(left, AstNode::Identifier(_)));
                                assert!(matches!(op, Comparator::Eq));
                                assert!(matches!(right, AstNode::Number(10)));
                            }
                            _ => panic!("Expected Comparison, got {:?}", ands[0]),
                        }
                    }
                    _ => panic!("Expected And, got {:?}", ors[0]),
                }
            }
            _ => panic!("Expected Or, got {:?}", ast),
        }
    }

    #[test]
    fn test_arena_parse_and_expression() {
        let parser = ArenaParser::new();
        let ast = parser.parse_rule("true AND false");

        // The grammar always wraps in Or([And([...])])
        match ast {
            AstNode::Or(ors) => {
                assert_eq!(ors.len(), 1);
                match &ors[0] {
                    AstNode::And(ands) => {
                        assert_eq!(ands.len(), 2);
                        assert!(matches!(ands[0], AstNode::Bool(true)));
                        assert!(matches!(ands[1], AstNode::Bool(false)));
                    }
                    _ => panic!("Expected And, got {:?}", ors[0]),
                }
            }
            _ => panic!("Expected Or, got {:?}", ast),
        }
    }

    #[test]
    fn test_arena_parse_or_expression() {
        let parser = ArenaParser::new();
        let ast = parser.parse_rule("true OR false");

        // For OR, it should wrap in Or with 2 And children
        match ast {
            AstNode::Or(ors) => {
                assert_eq!(ors.len(), 2);
                // Each branch is wrapped in And
                match &ors[0] {
                    AstNode::And(ands) => {
                        assert_eq!(ands.len(), 1);
                        assert!(matches!(ands[0], AstNode::Bool(true)));
                    }
                    _ => panic!("Expected And, got {:?}", ors[0]),
                }
                match &ors[1] {
                    AstNode::And(ands) => {
                        assert_eq!(ands.len(), 1);
                        assert!(matches!(ands[0], AstNode::Bool(false)));
                    }
                    _ => panic!("Expected And, got {:?}", ors[1]),
                }
            }
            _ => panic!("Expected Or, got {:?}", ast),
        }
    }

    #[test]
    fn test_arena_parse_list_literal() {
        let parser = ArenaParser::new();
        let ast = parser.parse_rule(r#"["a", "b", "c"]"#);

        // The grammar wraps in Or([And([...])])
        match ast {
            AstNode::Or(ors) => {
                assert_eq!(ors.len(), 1);
                match &ors[0] {
                    AstNode::And(ands) => {
                        assert_eq!(ands.len(), 1);
                        match &ands[0] {
                            AstNode::ListLiteral(elements) => {
                                assert_eq!(elements.len(), 3);
                            }
                            _ => panic!("Expected ListLiteral, got {:?}", ands[0]),
                        }
                    }
                    _ => panic!("Expected And, got {:?}", ors[0]),
                }
            }
            _ => panic!("Expected Or, got {:?}", ast),
        }
    }

    #[test]
    fn test_arena_parse_map_literal() {
        let parser = ArenaParser::new();
        let ast = parser.parse_rule(r#"{"key": "value"}"#);

        // The grammar wraps in Or([And([...])])
        match ast {
            AstNode::Or(ors) => {
                assert_eq!(ors.len(), 1);
                match &ors[0] {
                    AstNode::And(ands) => {
                        assert_eq!(ands.len(), 1);
                        match &ands[0] {
                            AstNode::MapLiteral(entries) => {
                                assert_eq!(entries.len(), 1);
                                assert_eq!(entries[0].0, "key");
                            }
                            _ => panic!("Expected MapLiteral, got {:?}", ands[0]),
                        }
                    }
                    _ => panic!("Expected And, got {:?}", ors[0]),
                }
            }
            _ => panic!("Expected Or, got {:?}", ast),
        }
    }

    #[test]
    fn test_arena_evaluate_simple_boolean() {
        let parser = ArenaParser::new();
        let ctx = FactsEvalContext::new();

        let result = evaluate_arena("true", &ctx, &parser).expect("eval failed");
        assert!(result);

        let result = evaluate_arena("false", &ctx, &parser).expect("eval failed");
        assert!(!result);
    }

    #[test]
    fn test_arena_evaluate_and() {
        let parser = ArenaParser::new();
        let ctx = FactsEvalContext::new();

        let result = evaluate_arena("true AND true", &ctx, &parser).expect("eval failed");
        assert!(result);

        let result = evaluate_arena("true AND false", &ctx, &parser).expect("eval failed");
        assert!(!result);
    }

    #[test]
    fn test_arena_evaluate_or() {
        let parser = ArenaParser::new();
        let ctx = FactsEvalContext::new();

        let result = evaluate_arena("true OR false", &ctx, &parser).expect("eval failed");
        assert!(result);

        let result = evaluate_arena("false OR false", &ctx, &parser).expect("eval failed");
        assert!(!result);
    }

    #[test]
    fn test_arena_evaluate_with_facts() {
        let parser = ArenaParser::new();
        let mut ctx = FactsEvalContext::new();
        ctx.add_fact("vars.x", Value::Number(10.0));
        ctx.add_fact("vars.y", Value::Number(20.0));

        let result = evaluate_arena(r#"vars.x == 10"#, &ctx, &parser);
        assert!(
            result.is_ok(),
            "vars.x == 10 failed with error: {:?}",
            result.err()
        );
        assert!(result.unwrap(), "vars.x == 10 should be true");

        let result = evaluate_arena(r#"vars.x < vars.y"#, &ctx, &parser);
        assert!(
            result.is_ok(),
            "vars.x < vars.y failed with error: {:?}",
            result.err()
        );
        assert!(result.unwrap(), "vars.x < vars.y should be true");

        let result = evaluate_arena(r#"vars.x > vars.y"#, &ctx, &parser);
        assert!(
            result.is_ok(),
            "vars.x > vars.y failed with error: {:?}",
            result.err()
        );
        assert!(!result.unwrap(), "vars.x > vars.y should be false");
    }

    #[test]
    fn test_arena_evaluate_complex_expression() {
        let parser = ArenaParser::new();
        let mut ctx = FactsEvalContext::new();
        ctx.add_fact("vars.x", Value::Number(10.0));
        ctx.add_fact("vars.y", Value::Number(20.0));
        ctx.add_fact("vars.z", Value::Number(30.0));

        let expr = r#"(vars.x < vars.y) AND (vars.y < vars.z)"#;
        let result = evaluate_arena(expr, &ctx, &parser).expect("eval failed");
        assert!(result);

        let expr = r#"(vars.x > vars.y) OR (vars.y < vars.z)"#;
        let result = evaluate_arena(expr, &ctx, &parser).expect("eval failed");
        assert!(result);
    }

    #[test]
    fn test_arena_parser_reset() {
        let mut parser = ArenaParser::new();
        let mut ctx = FactsEvalContext::new();
        ctx.add_fact("vars.x", Value::Number(10.0));

        let result1 = evaluate_arena(r#"vars.x == 10"#, &ctx, &parser).expect("eval failed");
        assert!(result1);

        parser.reset();

        let result2 = evaluate_arena(r#"vars.x > 5"#, &ctx, &parser).expect("eval failed");
        assert!(result2);
    }

    #[test]
    fn test_arena_string_comparison() {
        let parser = ArenaParser::new();
        let mut ctx = FactsEvalContext::new();
        ctx.add_fact("user.name", Value::String(Arc::from("Alice")));

        let result = evaluate_arena(r#"user.name == "Alice""#, &ctx, &parser).expect("eval failed");
        assert!(result);

        let result = evaluate_arena(r#"user.name != "Bob""#, &ctx, &parser).expect("eval failed");
        assert!(result);
    }

    #[test]
    fn test_arena_list_contains() {
        let parser = ArenaParser::new();
        let mut ctx = FactsEvalContext::new();
        ctx.add_fact(
            "data.items",
            Value::List(vec![
                Value::String(Arc::from("a")),
                Value::String(Arc::from("b")),
                Value::String(Arc::from("c")),
            ]),
        );

        let result =
            evaluate_arena(r#"data.items CONTAINS "b""#, &ctx, &parser).expect("eval failed");
        assert!(result);

        let result =
            evaluate_arena(r#"data.items CONTAINS "d""#, &ctx, &parser).expect("eval failed");
        assert!(!result);
    }

    #[test]
    fn test_arena_in_operator() {
        let parser = ArenaParser::new();
        let mut ctx = FactsEvalContext::new();
        ctx.add_fact("data.x", Value::String(Arc::from("b")));

        let result =
            evaluate_arena(r#"data.x IN ["a", "b", "c"]"#, &ctx, &parser).expect("eval failed");
        assert!(result);

        let result =
            evaluate_arena(r#"data.x IN ["d", "e", "f"]"#, &ctx, &parser).expect("eval failed");
        assert!(!result);
    }

    #[test]
    fn test_arena_heap_equivalence() {
        // Arena and heap parsers must agree on the result.
        let arena_parser = ArenaParser::new();
        let mut ctx = FactsEvalContext::new();
        ctx.add_fact("vars.x", Value::Number(42.0));
        ctx.add_fact("data.y", Value::String(Arc::from("test")));
        ctx.add_fact(
            "list.items",
            Value::List(vec![
                Value::Number(1.0),
                Value::Number(2.0),
                Value::Number(3.0),
            ]),
        );

        let test_cases = vec![
            r#"vars.x == 42"#,
            r#"vars.x > 40 AND vars.x < 50"#,
            r#"data.y == "test""#,
            r#"list.items CONTAINS 2"#,
            r#"(vars.x > 40) OR (data.y != "test")"#,
        ];

        for expr in test_cases {
            let arena_result = evaluate_arena(expr, &ctx, &arena_parser)
                .unwrap_or_else(|e| panic!("arena eval failed for {}: {}", expr, e));
            let heap_result = crate::evaluate(expr, &ctx)
                .unwrap_or_else(|e| panic!("heap eval failed for {}: {}", expr, e));

            assert_eq!(
                arena_result, heap_result,
                "Results differ for expression: {}",
                expr
            );
        }
    }
}
