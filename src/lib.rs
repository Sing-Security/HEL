use pest::iterators::Pair;
use pest::Parser;
use pest_derive::Parser;
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

#[derive(Parser)]
#[grammar = "hel.pest"]
pub struct HelParser;

#[derive(Debug, Clone)]
pub enum AstNode {
    Bool(bool),
    String(Arc<str>),
    Number(u64),
    /// Float number (f64)
    Float(f64),
    Identifier(Arc<str>),
    Attribute {
        object: Arc<str>,
        field: Arc<str>,
    },
    Comparison {
        left: Box<AstNode>,
        op: Comparator,
        right: Box<AstNode>,
    },
    And(Vec<AstNode>),
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Comparator {
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
    Contains,
    /// IN operator for membership tests (e.g., "a" IN ["a", "b"])
    In,
}

/// Runtime value type for HEL evaluation
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    String(Arc<str>),
    Number(f64),
    List(Vec<Value>),
    Map(BTreeMap<Arc<str>, Value>),
}

/// Resolver interface for host integration
///
/// Products implement this trait to provide values for attribute access
/// in HEL expressions. This decouples HEL from domain-specific facts.
pub trait HelResolver {
    /// Resolve an attribute path (object.field) to a value
    ///
    /// Returns `Some(Value)` if the attribute exists, `None` if missing.
    /// Missing attributes are treated as `Null` by the evaluator.
    fn resolve_attr(&self, object: &str, field: &str) -> Option<Value>;
}

/// Evaluation context that includes resolver and optional built-ins registry
pub struct EvalContext<'a> {
    resolver: &'a dyn HelResolver,
    builtins: Option<&'a builtins::BuiltinsRegistry>,
}

impl<'a> EvalContext<'a> {
    /// Create a context with just a resolver (no built-ins)
    pub fn new(resolver: &'a dyn HelResolver) -> Self {
        Self {
            resolver,
            builtins: None,
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
        }
    }
}

/// Error type for HEL evaluation
#[derive(Debug, Clone)]
pub enum EvalError {
    UnknownAttribute {
        object: String,
        field: String,
    },
    TypeMismatch {
        expected: String,
        got: String,
        context: String,
    },
    InvalidOperation(String),
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

pub fn parse_rule(input: &str) -> AstNode {
    let mut pairs = HelParser::parse(Rule::condition, input).expect("parse error");
    build_ast(pairs.next().unwrap())
}

fn build_ast(pair: Pair<Rule>) -> AstNode {
    match pair.as_rule() {
        Rule::condition => {
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

            // Check if second element exists (namespace.function case)
            let second = inner.next();
            let (namespace, name, remaining_args) = if second.is_some() {
                (
                    Some(Arc::from(first.as_str())),
                    Arc::from(second.unwrap().as_str()),
                    inner,
                )
            } else {
                (None, Arc::from(first.as_str()), inner)
            };

            // Parse arguments from remaining items
            let args: Vec<AstNode> = remaining_args.map(|arg| build_ast(arg)).collect();

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

/// New evaluation API (resolver-based)

pub fn evaluate_with_resolver(
    condition: &str,
    resolver: &dyn HelResolver,
) -> Result<bool, EvalError> {
    let ast = parse_rule(condition);
    let ctx = EvalContext::new(resolver);
    evaluate_ast_with_context(&ast, &ctx)
}

pub fn evaluate_with_context(
    condition: &str,
    resolver: &dyn HelResolver,
    builtins: &builtins::BuiltinsRegistry,
) -> Result<bool, EvalError> {
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
        _ => Ok(false),
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
        AstNode::Identifier(s) => Ok(Value::String(s.clone())),
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
        AstNode::FunctionCall {
            namespace,
            name,
            args,
        } => {
            // Evaluate arguments
            let arg_values: Result<Vec<Value>, EvalError> = args
                .iter()
                .map(|arg| eval_node_to_value_with_context(arg, ctx))
                .collect();
            let arg_values = arg_values?;

            // Call built-in function if registry is available
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
        _ => Ok(Value::Null),
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

#[cfg(test)]
mod tests {
    use super::*;

    // Basic resolver used in trace tests and other unit tests
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
}
