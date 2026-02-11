//! Arena-allocated AST for HEL
//!
//! This module provides an arena-based memory allocation strategy for HEL AST nodes.
//! Arena allocation improves performance by:
//!
//! 1. **Fast allocation**: O(1) bump pointer allocation vs global allocator overhead
//! 2. **Cache locality**: Nodes are adjacent in memory, improving CPU cache utilization
//! 3. **Batch deallocation**: Dropping the arena frees all nodes at once, no recursive Drop
//!
//! # When to use arena allocation
//!
//! Arena allocation is particularly beneficial when:
//!
//! - Parsing and evaluating many expressions in a tight loop (e.g., rule engines)
//! - Expression lifetime is known and bounded (e.g., single request processing)
//! - Memory pressure from many small heap allocations is a concern
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
//! For maximum performance when processing multiple expressions, reuse the same
//! `ArenaParser` instance and call `reset()` between expressions:
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
//! // Parse and evaluate first expression
//! let result1 = evaluate_arena(r#"vars.x == 10"#, &ctx, &parser).expect("eval failed");
//! assert!(result1);
//!
//! // Reset arena to reuse memory
//! parser.reset();
//!
//! // Parse and evaluate second expression
//! let result2 = evaluate_arena(r#"vars.x > 5"#, &ctx, &parser).expect("eval failed");
//! assert!(result2);
//! ```

use bumpalo::Bump;
use bumpalo::collections::Vec as BumpVec;

use crate::{
    Comparator, EvalError, FactsEvalContext, HelError, HelParser, HelResolver, Rule, Value,
    builtins::BuiltinsRegistry,
};
use pest::Parser;

// ============================================================================
// Arena AST Types
// ============================================================================

/// An arena-allocated AST node
///
/// Uses references into the arena (`&'arena`) instead of `Box` / `Vec` / `Arc`.
/// This eliminates individual heap allocations and improves cache locality.
///
/// The `'arena` lifetime ties all nodes to their arena — no use-after-free possible.
#[derive(Debug, Clone, Copy)]
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

/// Parser that builds arena-allocated ASTs
///
/// This parser allocates all AST nodes into a single contiguous memory region
/// (the arena). This improves parse speed and memory locality compared to
/// individual heap allocations.
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
    pub fn new() -> Self {
        Self {
            arena: Bump::new(),
        }
    }

    /// Parse a HEL rule into an arena-allocated AST
    ///
    /// # Arguments
    ///
    /// * `input` - The HEL expression string to parse
    ///
    /// # Returns
    ///
    /// Returns a reference to the parsed AST node, which borrows from `self`.
    /// The node is valid as long as the parser is not reset or dropped.
    ///
    /// # Panics
    ///
    /// Panics if the input fails to parse.
    pub fn parse_rule<'a>(&'a self, input: &str) -> &'a AstNode<'a> {
        let mut pairs = HelParser::parse(Rule::condition, input).expect("parse error");
        self.build_ast_arena(pairs.next().unwrap())
    }

    /// Parse a HEL expression with validation
    ///
    /// # Arguments
    ///
    /// * `expr` - The HEL expression string to parse
    ///
    /// # Returns
    ///
    /// Returns `Ok` with a reference to the parsed AST node, or `Err` if parsing fails.
    pub fn parse_expression<'a>(&'a self, expr: &str) -> Result<&'a AstNode<'a>, HelError> {
        crate::validate_expression(expr)?;
        Ok(self.parse_rule(expr))
    }

    /// Reset the arena for reuse
    ///
    /// This clears all allocations and allows the arena memory to be reused
    /// for subsequent parses. This is more efficient than creating a new parser.
    ///
    /// # Warning
    ///
    /// All AST nodes previously returned by this parser become invalid after reset.
    /// Using them will lead to undefined behavior.
    pub fn reset(&mut self) {
        self.arena.reset();
    }

    /// Build an arena-allocated AST from a pest Pair
    fn build_ast_arena<'a>(
        &'a self,
        pair: pest::iterators::Pair<Rule>,
    ) -> &'a AstNode<'a> {
        let node = match pair.as_rule() {
            Rule::condition => {
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
                    None => panic!("Failed to parse number literal: '{}'", num_str),
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
                        let value = *self.build_ast_arena(entry_inner.next().expect("Missing map value"));
                        entries.push((key, value));
                    }
                }
                AstNode::MapLiteral(entries.into_bump_slice())
            }

            Rule::function_call => {
                let mut inner = pair.into_inner();
                let first = inner.next().expect("Missing function name");

                // Check if second element exists (namespace.function case)
                let second = inner.next();
                let (namespace, name, remaining_args): (Option<&str>, &str, _) = if second.is_some() {
                    (
                        Some(self.arena.alloc_str(first.as_str())),
                        self.arena.alloc_str(second.unwrap().as_str()),
                        inner,
                    )
                } else {
                    (None, self.arena.alloc_str(first.as_str()), inner)
                };

                // Parse arguments from remaining items
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
/// This is a high-level convenience function that parses and evaluates
/// an expression using arena allocation.
///
/// # Arguments
///
/// * `expr` - The HEL expression string to evaluate
/// * `context` - Facts context providing attribute values
/// * `parser` - Arena parser (can be reused across calls)
///
/// # Returns
///
/// Returns `Ok(true)` if the expression evaluates to true, `Ok(false)` otherwise.
/// Returns `Err` if parsing or evaluation fails.
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
///
/// # Arguments
///
/// * `condition` - The HEL expression string to evaluate
/// * `resolver` - Implementation of `HelResolver` to provide attribute values
/// * `parser` - Arena parser
///
/// # Returns
///
/// Returns `Ok(true)` if the condition evaluates to true, `Ok(false)` otherwise.
pub fn evaluate_with_resolver_arena(
    condition: &str,
    resolver: &dyn HelResolver,
    parser: &ArenaParser,
) -> Result<bool, EvalError> {
    let ast = parser.parse_rule(condition);
    let ctx = ArenaEvalContext::new(resolver);
    evaluate_ast_arena(ast, &ctx)
}

/// Evaluate arena AST with resolver and builtins
///
/// Low-level API for evaluating an arena-allocated AST with a custom resolver
/// and built-in functions.
///
/// # Arguments
///
/// * `condition` - The HEL expression string to evaluate
/// * `resolver` - Implementation of `HelResolver` to provide attribute values
/// * `builtins` - Registry of built-in functions
/// * `parser` - Arena parser
///
/// # Returns
///
/// Returns `Ok(true)` if the condition evaluates to true, `Ok(false)` otherwise.
pub fn evaluate_with_context_arena(
    condition: &str,
    resolver: &dyn HelResolver,
    builtins: &BuiltinsRegistry,
    parser: &ArenaParser,
) -> Result<bool, EvalError> {
    let ast = parser.parse_rule(condition);
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
        AstNode::Comparison { left, op, right } => {
            evaluate_comparison_arena(left, *op, right, ctx)
        }
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
    use std::sync::Arc;
    use std::collections::BTreeMap;

    match node {
        AstNode::Bool(b) => Ok(Value::Bool(*b)),
        AstNode::String(s) => Ok(Value::String(Arc::from(*s))),
        AstNode::Number(n) => Ok(Value::Number(*n as f64)),
        AstNode::Float(f) => Ok(Value::Number(*f)),
        AstNode::Identifier(s) => {
            // In arena mode, we don't have variable bindings (yet)
            // so identifiers are treated as string literals
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
        // Handle boolean expressions (Comparison, And, Or)
        AstNode::Comparison { .. } | AstNode::And(_) | AstNode::Or(_) => {
            // Evaluate as boolean and wrap in Value::Bool
            let bool_result = evaluate_ast_arena(node, ctx)?;
            Ok(Value::Bool(bool_result))
        }
        AstNode::FunctionCall { namespace, name, args } => {
            // Evaluate arguments
            let arg_values: Result<Vec<Value>, EvalError> = args
                .iter()
                .map(|arg| eval_node_to_value_arena(arg, ctx))
                .collect();
            let arg_values = arg_values?;

            // Call built-in function if registry is available
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
        assert!(result.is_ok(), "vars.x == 10 failed with error: {:?}", result.err());
        assert!(result.unwrap(), "vars.x == 10 should be true");
        
        let result = evaluate_arena(r#"vars.x < vars.y"#, &ctx, &parser);
        assert!(result.is_ok(), "vars.x < vars.y failed with error: {:?}", result.err());
        assert!(result.unwrap(), "vars.x < vars.y should be true");
        
        let result = evaluate_arena(r#"vars.x > vars.y"#, &ctx, &parser);
        assert!(result.is_ok(), "vars.x > vars.y failed with error: {:?}", result.err());
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
        
        // First parse and evaluate
        let result1 = evaluate_arena(r#"vars.x == 10"#, &ctx, &parser).expect("eval failed");
        assert!(result1);
        
        // Reset arena
        parser.reset();
        
        // Second parse and evaluate
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
        ctx.add_fact("data.items", Value::List(vec![
            Value::String(Arc::from("a")),
            Value::String(Arc::from("b")),
            Value::String(Arc::from("c")),
        ]));
        
        let result = evaluate_arena(r#"data.items CONTAINS "b""#, &ctx, &parser).expect("eval failed");
        assert!(result);
        
        let result = evaluate_arena(r#"data.items CONTAINS "d""#, &ctx, &parser).expect("eval failed");
        assert!(!result);
    }

    #[test]
    fn test_arena_in_operator() {
        let parser = ArenaParser::new();
        let mut ctx = FactsEvalContext::new();
        ctx.add_fact("data.x", Value::String(Arc::from("b")));
        
        let result = evaluate_arena(r#"data.x IN ["a", "b", "c"]"#, &ctx, &parser).expect("eval failed");
        assert!(result);
        
        let result = evaluate_arena(r#"data.x IN ["d", "e", "f"]"#, &ctx, &parser).expect("eval failed");
        assert!(!result);
    }

    #[test]
    fn test_arena_heap_equivalence() {
        // Test that arena and heap parsers produce equivalent evaluation results
        let arena_parser = ArenaParser::new();
        let mut ctx = FactsEvalContext::new();
        ctx.add_fact("vars.x", Value::Number(42.0));
        ctx.add_fact("data.y", Value::String(Arc::from("test")));
        ctx.add_fact("list.items", Value::List(vec![
            Value::Number(1.0),
            Value::Number(2.0),
            Value::Number(3.0),
        ]));
        
        let test_cases = vec![
            r#"vars.x == 42"#,
            r#"vars.x > 40 AND vars.x < 50"#,
            r#"data.y == "test""#,
            r#"list.items CONTAINS 2"#,
            r#"(vars.x > 40) OR (data.y != "test")"#,
        ];
        
        for expr in test_cases {
            let arena_result = evaluate_arena(expr, &ctx, &arena_parser)
                .expect(&format!("arena eval failed for: {}", expr));
            let heap_result = crate::evaluate(expr, &ctx)
                .expect(&format!("heap eval failed for: {}", expr));
            
            assert_eq!(
                arena_result, heap_result,
                "Results differ for expression: {}",
                expr
            );
        }
    }
}
