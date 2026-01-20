//! Trace capture for HEL rule evaluation
//!
//! This module provides evaluation tracing to explain why a rule matched or didn't match.
//! It captures atom-level comparisons with resolved values for deterministic audit trails.

use crate::{AstNode, Comparator, EvalContext, EvalError, Value};

/// Trace of a single comparison atom in a rule
#[derive(Debug, Clone)]
pub struct AtomTrace {
    /// Left side of comparison (as string)
    pub left: String,

    /// Comparison operator
    pub op: Comparator,

    /// Right side of comparison (as string)
    pub right: String,

    /// Resolved value from the left side
    pub resolved_left_value: Option<String>,

    /// Resolved value from the right side
    pub resolved_right_value: Option<String>,

    /// Result of this atom evaluation
    pub atom_result: bool,
}

/// Complete evaluation trace for a rule
#[derive(Debug, Clone)]
pub struct EvalTrace {
    /// Final result of evaluation
    pub result: bool,

    /// Atom-level traces (in evaluation order)
    pub atoms: Vec<AtomTrace>,

    /// Fact paths that were accessed during evaluation (stored as HashSet internally)
    facts_used_set: std::collections::HashSet<String>,
}

impl EvalTrace {
    /// Create a new empty trace
    pub fn new() -> Self {
        Self {
            result: false,
            atoms: Vec::new(),
            facts_used_set: std::collections::HashSet::new(),
        }
    }

    /// Add an atom trace
    pub fn add_atom(&mut self, atom: AtomTrace) {
        // Track fact paths from left side (attributes)
        if atom.left.contains('.') {
            self.facts_used_set.insert(atom.left.clone());
        }

        self.atoms.push(atom);
    }

    /// Set the final result
    pub fn set_result(&mut self, result: bool) {
        self.result = result;
    }

    /// Get facts used (sorted for determinism)
    pub fn facts_used(&self) -> Vec<String> {
        let mut facts: Vec<String> = self.facts_used_set.iter().cloned().collect();
        facts.sort();
        facts
    }
}

impl Default for EvalTrace {
    fn default() -> Self {
        Self::new()
    }
}

/// Evaluate a condition with tracing enabled
///
/// This function evaluates the condition and captures a detailed trace showing
/// which atoms were evaluated, what values they resolved to, and what the results were.
pub fn evaluate_with_trace(
    condition: &str,
    resolver: &dyn crate::HelResolver,
    builtins: Option<&crate::builtins::BuiltinsRegistry>,
) -> Result<EvalTrace, EvalError> {
    let ast = crate::parse_rule(condition);
    let ctx = if let Some(b) = builtins {
        EvalContext::with_builtins(resolver, b)
    } else {
        EvalContext::new(resolver)
    };

    let mut trace = EvalTrace::new();
    let result = evaluate_ast_with_trace(&ast, &ctx, &mut trace)?;
    trace.set_result(result);

    Ok(trace)
}

/// Evaluate AST node with trace capture
fn evaluate_ast_with_trace(
    ast: &AstNode,
    ctx: &EvalContext,
    trace: &mut EvalTrace,
) -> Result<bool, EvalError> {
    match ast {
        AstNode::Bool(b) => Ok(*b),
        AstNode::And(nodes) => {
            for node in nodes {
                if !evaluate_ast_with_trace(node, ctx, trace)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        AstNode::Or(nodes) => {
            for node in nodes {
                if evaluate_ast_with_trace(node, ctx, trace)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        AstNode::Comparison { left, op, right } => {
            evaluate_comparison_with_trace(left, *op, right, ctx, trace)
        }
        _ => Ok(false),
    }
}

/// Evaluate a comparison with trace capture
fn evaluate_comparison_with_trace(
    left: &AstNode,
    op: Comparator,
    right: &AstNode,
    ctx: &EvalContext,
    trace: &mut EvalTrace,
) -> Result<bool, EvalError> {
    // Evaluate left and right nodes
    let left_val = eval_node_to_value_with_context(left, ctx)?;
    let right_val = eval_node_to_value_with_context(right, ctx)?;

    // Perform comparison
    let result = crate::compare_new_values(&left_val, &right_val, op);

    // Record atom trace
    let atom = AtomTrace {
        left: node_to_string(left),
        op,
        right: node_to_string(right),
        resolved_left_value: Some(value_to_string(&left_val)),
        resolved_right_value: Some(value_to_string(&right_val)),
        atom_result: result,
    };

    trace.add_atom(atom);

    Ok(result)
}

/// Convert an AST node to a string representation
fn node_to_string(node: &AstNode) -> String {
    match node {
        AstNode::Bool(b) => b.to_string(),
        AstNode::String(s) => format!("\"{}\"", s),
        AstNode::Number(n) => n.to_string(),
        AstNode::Float(f) => f.to_string(),
        AstNode::Identifier(s) => s.to_string(),
        AstNode::Attribute { object, field } => format!("{}.{}", object, field),
        AstNode::ListLiteral(_) => "[...]".to_string(),
        AstNode::MapLiteral(_) => "{...}".to_string(),
        AstNode::FunctionCall {
            namespace, name, ..
        } => {
            if let Some(ns) = namespace {
                format!("{}.{}(...)", ns, name)
            } else {
                format!("{}(...)", name)
            }
        }
        _ => "?".to_string(),
    }
}

/// Convert a Value to a string representation
fn value_to_string(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::String(s) => s.to_string(),
        Value::Number(n) => n.to_string(),
        Value::List(items) => {
            let strs: Vec<String> = items.iter().map(value_to_string).collect();
            format!("[{}]", strs.join(", "))
        }
        Value::Map(m) => {
            let entries: Vec<String> = m
                .iter()
                .map(|(k, v)| format!("{}: {}", k, value_to_string(v)))
                .collect();
            format!("{{{}}}", entries.join(", "))
        }
    }
}

/// Helper: return a stable textual operator for a `Comparator`.
fn comparator_to_str(op: Comparator) -> &'static str {
    match op {
        Comparator::Eq => "==",
        Comparator::Ne => "!=",
        Comparator::Gt => ">",
        Comparator::Ge => ">=",
        Comparator::Lt => "<",
        Comparator::Le => "<=",
        Comparator::Contains => "CONTAINS",
        Comparator::In => "IN",
    }
}

use std::fmt;

/// Pretty-print a single atom trace (stable, deterministic)
impl fmt::Display for AtomTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} {} => left_resolved={:?}, right_resolved={:?}, atom_result={}",
            self.left,
            comparator_to_str(self.op),
            self.right,
            self.resolved_left_value,
            self.resolved_right_value,
            self.atom_result
        )
    }
}

/// Pretty-print an EvalTrace as multi-line human-friendly output.
/// Hosts and examples can call `trace.to_string()` or `trace.pretty_print()` to
/// obtain deterministic, audit-friendly summaries.
impl fmt::Display for EvalTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Top-line: result
        writeln!(f, "Result: {}", self.result)?;
        // Atoms in order
        for (i, atom) in self.atoms.iter().enumerate() {
            writeln!(f, "  {}: {}", i, atom)?;
        }
        // Facts used summary (sorted)
        let facts = self.facts_used();
        if !facts.is_empty() {
            writeln!(f, "Facts used: {:?}", facts)?;
        }
        Ok(())
    }
}

/// Convenience method to get a pretty-printed string (avoids allocations for simple prints)
impl EvalTrace {
    /// Return a human-friendly, deterministic multi-line string of the trace.
    pub fn pretty_print(&self) -> String {
        use std::fmt::Write as FmtWrite;
        let mut out = String::new();
        let _ = write!(&mut out, "{}", self); // uses Display impl above
        out
    }
}

/// Re-export eval_node_to_value_with_context from parent module
/// (We need this for trace evaluation)
fn eval_node_to_value_with_context(node: &AstNode, ctx: &EvalContext) -> Result<Value, EvalError> {
    crate::eval_node_to_value_with_context(node, ctx)
}

// region:    --- Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HelResolver, Value};

    struct TestResolver;

    impl HelResolver for TestResolver {
        fn resolve_attr(&self, object: &str, field: &str) -> Option<Value> {
            match (object, field) {
                ("binary", "format") => Some(Value::String("elf".into())),
                ("security", "nx_enabled") => Some(Value::Bool(true)),
                _ => None,
            }
        }
    }

    #[test]
    fn test_evaluate_with_trace_simple() {
        let resolver = TestResolver;
        let condition = r#"binary.format == "elf""#;

        let trace = evaluate_with_trace(condition, &resolver, None).expect("evaluation failed");

        assert!(trace.result, "Condition should evaluate to true");
        assert_eq!(trace.atoms.len(), 1, "Should have one atom");
        assert_eq!(trace.atoms[0].left, "binary.format");
        assert_eq!(trace.atoms[0].right, "\"elf\"");
        assert_eq!(trace.atoms[0].resolved_left_value, Some("elf".to_string()));
        assert_eq!(trace.atoms[0].resolved_right_value, Some("elf".to_string()));
        assert!(trace.atoms[0].atom_result);
    }

    #[test]
    fn test_evaluate_with_trace_and() {
        let resolver = TestResolver;
        let condition = r#"binary.format == "elf" AND security.nx_enabled == true"#;

        let trace = evaluate_with_trace(condition, &resolver, None).expect("evaluation failed");

        assert!(trace.result, "Condition should evaluate to true");
        assert_eq!(trace.atoms.len(), 2, "Should have two atoms");
        assert!(trace.atoms[0].atom_result);
        assert!(trace.atoms[1].atom_result);
    }

    #[test]
    fn test_evaluate_with_trace_false_result() {
        let resolver = TestResolver;
        let condition = r#"binary.format == "pe""#;

        let trace = evaluate_with_trace(condition, &resolver, None).expect("evaluation failed");

        assert!(!trace.result, "Condition should evaluate to false");
        assert_eq!(trace.atoms.len(), 1, "Should have one atom");
        assert_eq!(trace.atoms[0].resolved_left_value, Some("elf".to_string()));
        assert_eq!(trace.atoms[0].resolved_right_value, Some("pe".to_string()));
        assert!(!trace.atoms[0].atom_result);
    }

    #[test]
    fn test_trace_facts_used() {
        let resolver = TestResolver;
        let condition = r#"binary.format == "elf" AND security.nx_enabled == true"#;

        let trace = evaluate_with_trace(condition, &resolver, None).expect("evaluation failed");

        let facts_used = trace.facts_used();
        assert!(facts_used.contains(&"binary.format".to_string()));
        assert!(facts_used.contains(&"security.nx_enabled".to_string()));

        // Should be sorted for determinism
        assert_eq!(facts_used[0], "binary.format");
        assert_eq!(facts_used[1], "security.nx_enabled");
    }
}

// endregion: --- Tests
