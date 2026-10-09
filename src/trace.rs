//! Trace capture for HEL rule evaluation.
//!
//! Records per-atom comparisons with their resolved values, so a rule's match or non-match can
//! be explained after the fact.

use std::fmt;

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

    /// Fact paths read during evaluation. Kept as a set to deduplicate; read them back
    /// sorted via [`EvalTrace::facts_used`].
    facts_used_set: std::collections::HashSet<String>,
}

impl EvalTrace {
    /// Create a new empty trace
    #[must_use]
    pub fn new() -> Self {
        Self {
            result: false,
            atoms: Vec::new(),
            facts_used_set: std::collections::HashSet::new(),
        }
    }

    /// Add an atom trace
    ///
    /// Fact paths are not taken from the atom's text: they are collected from the AST while
    /// the expression is evaluated (see [`EvalTrace::facts_used`]), so an atom added by hand
    /// does not contribute to `facts_used`.
    pub fn add_atom(&mut self, atom: AtomTrace) {
        self.atoms.push(atom);
    }

    /// Set the final result
    pub fn set_result(&mut self, result: bool) {
        self.result = result;
    }

    /// Get facts used (sorted for determinism)
    ///
    /// The `object.field` attributes the evaluation actually read: those on either side of
    /// a comparison and those inside function-call arguments and list/map literals. A
    /// function name is never a fact path, and an attribute in an `AND`/`OR` branch that
    /// short-circuited is not reported, because it was never read.
    #[must_use]
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
/// Evaluates `condition` and records, for each atom in the expression, the value it
/// resolved to and the result it produced. Pass a `builtins` registry when the
/// expression calls functions; pass `None` otherwise. The returned [`EvalTrace`]
/// carries the final result, the ordered atoms, and the sorted set of facts read.
///
/// # Errors
///
/// Returns [`EvalError::ParseError`] if `condition` is not a valid HEL expression,
/// [`EvalError::InvalidOperation`] if the expression calls a function that `builtins`
/// does not define - or if `builtins` is `None` and the expression calls one at all -
/// and [`EvalError::TypeMismatch`] if an operand has the wrong type for its operator.
///
/// An attribute the resolver answers `None` for is not an error: it resolves to
/// [`Value::Null`] and appears in the trace as such.
///
/// # Examples
///
/// ```
/// use hel::trace::evaluate_with_trace;
/// use hel::{HelResolver, Value};
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
/// let trace = evaluate_with_trace(r#"binary.arch == "x86_64""#, &MyResolver, None)
///     .expect("evaluation failed");
/// assert!(trace.result);
/// assert_eq!(trace.facts_used(), vec!["binary.arch".to_string()]);
/// ```
pub fn evaluate_with_trace(
    condition: &str,
    resolver: &dyn crate::HelResolver,
    builtins: Option<&crate::builtins::BuiltinsRegistry>,
) -> Result<EvalTrace, EvalError> {
    crate::validate_expression(condition).map_err(|e| EvalError::ParseError(e.to_string()))?;
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
pub(crate) fn evaluate_ast_with_trace(
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
        // Any other node is a value rather than a condition, so it is only usable as a
        // condition when that value is itself a boolean. The same rule the ordinary
        // evaluator applies, so traced and untraced evaluation cannot disagree.
        other => {
            let value = crate::eval_node_to_value_with_context(other, ctx, Some(trace))?;
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

/// Evaluate a comparison with trace capture
fn evaluate_comparison_with_trace(
    left: &AstNode,
    op: Comparator,
    right: &AstNode,
    ctx: &EvalContext,
    trace: &mut EvalTrace,
) -> Result<bool, EvalError> {
    let left_val = crate::eval_node_to_value_with_context(left, ctx, Some(&mut *trace))?;
    let right_val = crate::eval_node_to_value_with_context(right, ctx, Some(&mut *trace))?;

    let result = crate::compare_new_values(&left_val, &right_val, op);

    // Fact paths come from the AST - both sides of the comparison, including attributes
    // inside arguments and literals - not from the atoms' display strings.
    collect_fact_paths(left, &mut trace.facts_used_set);
    collect_fact_paths(right, &mut trace.facts_used_set);

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

/// Collect the fact paths (`object.field` attributes) in a value-position AST subtree:
/// either side of a comparison, list/map literal elements and function-call arguments.
/// Conditions (`Comparison`/`And`/`Or`) stop the walk - they are evaluated through
/// [`evaluate_ast_with_trace`], which records their facts as they are actually reached, so
/// an `AND`/`OR` branch that short-circuits is not reported.
fn collect_fact_paths(node: &AstNode, out: &mut std::collections::HashSet<String>) {
    match node {
        AstNode::Attribute { object, field } => {
            out.insert(format!("{}.{}", object, field));
        }
        AstNode::ListLiteral(elements) => {
            for element in elements {
                collect_fact_paths(element, out);
            }
        }
        AstNode::MapLiteral(entries) => {
            for (_, value) in entries {
                collect_fact_paths(value, out);
            }
        }
        AstNode::FunctionCall { args, .. } => {
            for arg in args {
                collect_fact_paths(arg, out);
            }
        }
        _ => {}
    }
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

// Stable textual form of a `Comparator`, used by both `Display` impls below.
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

// Single-line, stable rendering of one comparison atom.
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

// Multi-line, human-readable rendering of a whole trace.
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

impl EvalTrace {
    /// A human-readable, deterministic multi-line rendering of the trace.
    ///
    /// Equivalent to `self.to_string()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hel::trace::{AtomTrace, EvalTrace};
    /// use hel::Comparator;
    ///
    /// let mut trace = EvalTrace::new();
    /// trace.add_atom(AtomTrace {
    ///     left: "binary.arch".to_string(),
    ///     op: Comparator::Eq,
    ///     right: "\"x86_64\"".to_string(),
    ///     resolved_left_value: Some("x86_64".to_string()),
    ///     resolved_right_value: Some("x86_64".to_string()),
    ///     atom_result: true,
    /// });
    /// trace.set_result(true);
    ///
    /// let text = trace.pretty_print();
    /// assert!(text.contains("Result: true"));
    /// assert!(text.contains("binary.arch == \"x86_64\""));
    /// ```
    #[must_use]
    pub fn pretty_print(&self) -> String {
        use std::fmt::Write as FmtWrite;
        let mut out = String::new();
        let _ = write!(&mut out, "{}", self);
        out
    }
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

    /// The traced evaluator must agree with the ordinary one: a standalone boolean call is
    /// evaluated through the registry, not silently reported as false.
    #[test]
    fn test_trace_standalone_call_matches_plain_evaluation() {
        let mut registry = crate::builtins::BuiltinsRegistry::new();
        registry
            .register(&crate::builtins::CoreBuiltinsProvider)
            .expect("register failed");

        let conditions = [
            r#"core.contains(["a"], "a")"#,
            r#"core.contains(["a"], "b")"#,
            "core.is_null(missing.attr)",
            r#"(binary.format == "elf") == true"#,
        ];

        for condition in conditions {
            let plain = crate::evaluate_with_context(condition, &TestResolver, &registry)
                .unwrap_or_else(|e| panic!("plain eval failed for {}: {:?}", condition, e));
            let traced = evaluate_with_trace(condition, &TestResolver, Some(&registry))
                .unwrap_or_else(|e| panic!("traced eval failed for {}: {:?}", condition, e));
            assert_eq!(plain, traced.result, "{}", condition);
        }
    }

    /// A value that is not a boolean is a type error on both paths, not a silent false.
    #[test]
    fn test_trace_non_boolean_condition_errors_like_plain() {
        let mut registry = crate::builtins::BuiltinsRegistry::new();
        registry
            .register(&crate::builtins::CoreBuiltinsProvider)
            .expect("register failed");

        for condition in [r#"core.len(["a"])"#, r#""just a string""#] {
            let plain = crate::evaluate_with_context(condition, &TestResolver, &registry);
            let traced = evaluate_with_trace(condition, &TestResolver, Some(&registry));
            assert!(
                matches!(plain, Err(EvalError::TypeMismatch { .. })),
                "plain {:?} for {}",
                plain,
                condition
            );
            assert!(
                matches!(traced, Err(EvalError::TypeMismatch { .. })),
                "traced {:?} for {}",
                traced,
                condition
            );
        }
    }

    /// Facts are collected from the AST: both sides of a comparison and inside call
    /// arguments, and never a function name.
    #[test]
    fn test_trace_facts_used_from_both_sides_and_args() {
        let mut registry = crate::builtins::BuiltinsRegistry::new();
        registry
            .register(&crate::builtins::CoreBuiltinsProvider)
            .expect("register failed");

        let trace = evaluate_with_trace("a.x == b.y", &TestResolver, None).expect("evaluated");
        assert_eq!(
            trace.facts_used(),
            vec!["a.x".to_string(), "b.y".to_string()]
        );

        // An attribute inside a call argument is a real dependency.
        let trace = evaluate_with_trace(
            r#"core.is_null(security.nx) == false"#,
            &TestResolver,
            Some(&registry),
        )
        .expect("evaluated");
        assert_eq!(trace.facts_used(), vec!["security.nx".to_string()]);

        // A call with no attribute anywhere reads no facts at all.
        let trace = evaluate_with_trace(
            r#"core.len(["a", "b"]) == 2"#,
            &TestResolver,
            Some(&registry),
        )
        .expect("evaluated");
        assert!(trace.facts_used().is_empty());

        // A nested condition's attributes are recorded too.
        let trace = evaluate_with_trace(r#"(binary.format == "elf") == true"#, &TestResolver, None)
            .expect("evaluated");
        assert_eq!(trace.facts_used(), vec!["binary.format".to_string()]);
        assert_eq!(trace.atoms.len(), 2, "nested condition records its atom");
    }
}

// endregion: --- Tests
