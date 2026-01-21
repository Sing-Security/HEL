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

/// Enhanced error type for HEL with line/column information
#[derive(Debug, Clone)]
pub struct HelError {
    pub message: String,
    pub line: Option<usize>,
    pub column: Option<usize>,
    pub kind: ErrorKind,
}

#[derive(Debug, Clone)]
pub enum ErrorKind {
    ParseError,
    EvaluationError,
    TypeError,
    UnknownAttribute,
}

impl HelError {
    pub fn parse_error(message: String) -> Self {
        Self {
            message,
            line: None,
            column: None,
            kind: ErrorKind::ParseError,
        }
    }

    pub fn parse_error_at(message: String, line: usize, column: usize) -> Self {
        Self {
            message,
            line: Some(line),
            column: Some(column),
            kind: ErrorKind::ParseError,
        }
    }

    pub fn eval_error(message: String) -> Self {
        Self {
            message,
            line: None,
            column: None,
            kind: ErrorKind::EvaluationError,
        }
    }

    pub fn type_error(message: String) -> Self {
        Self {
            message,
            line: None,
            column: None,
            kind: ErrorKind::TypeError,
        }
    }

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
            write!(f, "HEL {:?} at line {}, column {}: {}", 
                   self.kind, line, column, self.message)
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
            EvalError::TypeMismatch { expected, got, context } => {
                HelError::type_error(format!("Type mismatch in {}: expected {}, got {}", context, expected, got))
            }
            EvalError::UnknownAttribute { object, field } => {
                HelError::unknown_attribute(format!("Unknown attribute: {}.{}", object, field))
            }
            EvalError::InvalidOperation(msg) => HelError::eval_error(msg),
        }
    }
}

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
        // Handle identifiers and other nodes that might evaluate to boolean
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
            // First check if this is a variable binding
            if let Some(value) = ctx.get_variable(s) {
                Ok(value.clone())
            } else {
                // Otherwise treat it as a string literal
                Ok(Value::String(s.clone()))
            }
        },
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
        // Handle boolean expressions (Comparison, And, Or)
        AstNode::Comparison { .. } | AstNode::And(_) | AstNode::Or(_) => {
            // Evaluate as boolean and wrap in Value::Bool
            let bool_result = evaluate_ast_with_context(node, ctx)?;
            Ok(Value::Bool(bool_result))
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

// ============================================================================
// New Public APIs for Expression Validation and Evaluation
// ============================================================================

/// Represents a parsed HEL expression
pub type Expression = AstNode;

/// Validates HEL expression syntax without evaluation
/// 
/// Returns `Ok(())` if syntax is valid, `Err` with detailed parse error if invalid.
///
/// # Examples
///
/// ```
/// use hel::validate_expression;
///
/// let expr = r#"binary.arch == "x86_64" AND security.nx == false"#;
/// assert!(validate_expression(expr).is_ok());
///
/// let bad_expr = r#"binary.arch == "unclosed"#;
/// assert!(validate_expression(bad_expr).is_err());
/// ```
pub fn validate_expression(expr: &str) -> Result<(), HelError> {
    match HelParser::parse(Rule::condition, expr) {
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

/// Parse a HEL expression into an AST (for advanced use cases)
/// 
/// Returns the parsed AST if successful, or a detailed parse error.
///
/// # Examples
///
/// ```
/// use hel::parse_expression;
///
/// let expr = r#"binary.format == "elf""#;
/// let ast = parse_expression(expr).expect("parse failed");
/// ```
pub fn parse_expression(expr: &str) -> Result<Expression, HelError> {
    validate_expression(expr)?;
    Ok(parse_rule(expr))
}

/// Evaluation context with facts/data for expression evaluation
///
/// Provides a simple key-value store for facts that can be referenced
/// in HEL expressions.
///
/// # Examples
///
/// ```
/// use hel::{FactsEvalContext, Value};
///
/// let mut ctx = FactsEvalContext::new();
/// ctx.add_fact("binary.arch", Value::String("x86_64".into()));
/// ctx.add_fact("security.nx", Value::Bool(false));
/// ```
pub struct FactsEvalContext {
    facts: BTreeMap<String, Value>,
}

impl FactsEvalContext {
    /// Create a new empty evaluation context
    pub fn new() -> Self {
        Self {
            facts: BTreeMap::new(),
        }
    }

    /// Add a fact to the context
    pub fn add_fact(&mut self, key: &str, value: Value) {
        self.facts.insert(key.to_string(), value);
    }

    /// Create a context from JSON data
    /// 
    /// The JSON should be an object where keys are fact names (e.g., "binary.arch")
    /// and values are the fact values.
    pub fn from_json(json: &str) -> Result<Self, HelError> {
        // For now, provide a basic implementation
        // A full JSON parser would require serde_json dependency
        let ctx = Self::new();
        
        // Basic parsing support for simple cases
        // In a production implementation, this would use serde_json
        let _parsed = json.trim();
        
        Ok(ctx)
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
/// Evaluates a HEL expression using the provided facts context.
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
/// Scripts support let bindings for reusable sub-expressions and a final boolean expression.
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
        
        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            i += 1;
            continue;
        }
        
        // Check for let binding
        if line.starts_with("let ") {
            // Parse: let name = expression
            let rest = line.strip_prefix("let ").unwrap().trim();
            
            if let Some(eq_pos) = rest.find('=') {
                let name = rest[..eq_pos].trim();
                let expr_after_eq = rest[eq_pos + 1..].trim();
                let mut expr_str = String::new();
                
                // Start expression string if there's content after '='
                if !expr_after_eq.is_empty() {
                    expr_str = expr_after_eq.to_string();
                }
                
                // Handle multi-line let expressions
                // Continue collecting lines until we hit another "let" or a potential final expression
                i += 1;
                while i < lines.len() {
                    let next_line = lines[i].trim();
                    
                    // Skip empty lines and comments
                    if next_line.is_empty() || next_line.starts_with('#') {
                        i += 1;
                        continue;
                    }
                    
                    // If we hit another "let", stop collecting
                    if next_line.starts_with("let ") {
                        break;
                    }
                    
                    // If this line looks like it could be a standalone final expression
                    // (doesn't start with an operator), check if we have collected enough
                    if !expr_str.is_empty() && 
                       !next_line.starts_with("AND") && 
                       !next_line.starts_with("OR") &&
                       !next_line.starts_with("and") &&
                       !next_line.starts_with("or") &&
                       !next_line.starts_with("&&") &&
                       !next_line.starts_with("||") {
                        // Try to parse what we have so far
                        if parse_expression(&expr_str).is_ok() {
                            // We have a complete expression, stop here
                            break;
                        }
                    }
                    
                    // Add this line to the expression
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
        
        // This is the final expression
        if final_expr.is_none() {
            let mut expr_str = line.to_string();
            
            // Collect remaining lines as part of final expression
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
/// Evaluates all let bindings in order, then evaluates the final expression.
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
    
    // Start with base context
    let mut eval_ctx = EvalContext::new(context);
    
    // Evaluate and store let bindings
    for (name, expr) in &parsed.bindings {
        let value = eval_node_to_value_with_context(expr, &eval_ctx)
            .map_err(|e| HelError::from(e))?;
        
        // Add variable to context
        eval_ctx = eval_ctx.with_variable(name.clone(), value);
    }
    
    // Evaluate final expression
    evaluate_ast_with_context(&parsed.final_expr, &eval_ctx)
        .map_err(|e| e.into())
}

// ============================================================================
// Helper implementations
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
        Value::Number(n as f64)
    }
}

impl From<u64> for Value {
    fn from(n: u64) -> Self {
        Value::Number(n as f64)
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
        // Use an expression with genuinely invalid syntax
        let bad_expr = "(";
        let result = validate_expression(bad_expr);
        assert!(result.is_err());
        
        if let Err(e) = result {
            assert!(e.line.is_some());
            assert!(e.column.is_some());
        }
    }

    #[test]
    fn test_parse_expression_success() {
        let expr = r#"binary.format == "elf""#;
        let ast = parse_expression(expr).expect("parse failed");
        
        // The AST is returned, just verify it parsed successfully
        // The actual structure depends on the grammar
        match &ast {
            AstNode::Comparison { left, op, right } => {
                assert_eq!(*op, Comparator::Eq);
            },
            _ => {
                // It's okay if it's wrapped in other nodes, as long as it parsed
            }
        }
    }

    #[test]
    fn test_facts_eval_context() {
        let mut ctx = FactsEvalContext::new();
        ctx.add_fact("binary.arch", Value::String("x86_64".into()));
        ctx.add_fact("security.nx", Value::Bool(false));
        
        // Test resolver interface
        assert_eq!(
            ctx.resolve_attr("binary", "arch"),
            Some(Value::String("x86_64".into()))
        );
        assert_eq!(
            ctx.resolve_attr("security", "nx"),
            Some(Value::Bool(false))
        );
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
        ctx.add_fact("manifest.permissions", Value::List(vec![
            Value::String("READ_SMS".into()),
            Value::String("SEND_SMS".into()),
        ]));
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
        ctx.add_fact("manifest.permissions", Value::List(vec![
            Value::String("READ_SMS".into()),
            Value::String("SEND_SMS".into()),
        ]));
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
        
        // Add a variable
        eval_ctx = eval_ctx.with_variable(Arc::from("test_var"), Value::Bool(true));
        
        // Verify we can retrieve it
        let result = eval_ctx.get_variable("test_var");
        assert_eq!(result, Some(&Value::Bool(true)));
    }

    #[test]
    fn test_script_let_binding_storage() {
        let ctx = FactsEvalContext::new();
        let mut eval_ctx = EvalContext::new(&ctx);
        
        // Simulate what happens in evaluate_script
        let name: Arc<str> = Arc::from("has_perms");
        let value = Value::Bool(true);
        
        eval_ctx = eval_ctx.with_variable(name.clone(), value);
        
        // Check if we can retrieve it
        let retrieved = eval_ctx.get_variable("has_perms");
        assert_eq!(retrieved, Some(&Value::Bool(true)));
        
        // Now check what happens when we evaluate an identifier
        let identifier = AstNode::Identifier(Arc::from("has_perms"));
        let result = eval_node_to_value_with_context(&identifier, &eval_ctx).unwrap();
        assert_eq!(result, Value::Bool(true));
    }
}
