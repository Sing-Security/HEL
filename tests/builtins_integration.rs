//! Integration tests for HEL built-ins
//!
//! These tests demonstrate using built-in functions in HEL expressions.

use hel::{evaluate_with_context, BuiltinsRegistry, CoreBuiltinsProvider, BuiltinsProvider, HelResolver, Value};
use std::collections::BTreeMap;
use std::sync::Arc;

// Empty resolver for tests that only use literals and function calls
struct EmptyResolver;
impl HelResolver for EmptyResolver {
	fn resolve_attr(&self, _object: &str, _field: &str) -> Option<Value> {
		None
	}
}

#[test]
fn test_core_len_function_call() {
	let resolver = EmptyResolver;
	let mut registry = BuiltinsRegistry::new();
	let provider = CoreBuiltinsProvider;
	registry.register(&provider).expect("registration failed");

	// Test: core.len(["a", "b", "c"]) == 3
	let condition = r#"core.len(["a", "b", "c"]) == 3"#;
	let result = evaluate_with_context(condition, &resolver, &registry).expect("evaluation failed");
	assert!(result, "core.len should return 3 for list of 3 elements");
}

#[test]
fn test_core_contains_function_call() {
	let resolver = EmptyResolver;
	let mut registry = BuiltinsRegistry::new();
	let provider = CoreBuiltinsProvider;
	registry.register(&provider).expect("registration failed");

	// Test: core.contains(["a", "b", "c"], "b") == true
	let condition = r#"core.contains(["a", "b", "c"], "b") == true"#;
	let result = evaluate_with_context(condition, &resolver, &registry).expect("evaluation failed");
	assert!(result, "core.contains should find 'b' in list");

	// Test: core.contains(["a", "b", "c"], "d") == false
	let condition = r#"core.contains(["a", "b", "c"], "d") == false"#;
	let result = evaluate_with_context(condition, &resolver, &registry).expect("evaluation failed");
	assert!(result, "core.contains should not find 'd' in list");
}

#[test]
fn test_core_upper_lower_function_calls() {
	let resolver = EmptyResolver;
	let mut registry = BuiltinsRegistry::new();
	let provider = CoreBuiltinsProvider;
	registry.register(&provider).expect("registration failed");

	// Test: core.upper("hello") == "HELLO"
	let condition = r#"core.upper("hello") == "HELLO""#;
	let result = evaluate_with_context(condition, &resolver, &registry).expect("evaluation failed");
	assert!(result, "core.upper should convert to uppercase");

	// Test: core.lower("WORLD") == "world"
	let condition = r#"core.lower("WORLD") == "world""#;
	let result = evaluate_with_context(condition, &resolver, &registry).expect("evaluation failed");
	assert!(result, "core.lower should convert to lowercase");
}

#[test]
fn test_custom_domain_builtin() {
	struct TestResolver;
	impl HelResolver for TestResolver {
		fn resolve_attr(&self, object: &str, field: &str) -> Option<Value> {
			if object == "binary" && field == "format" {
				Some(Value::String("ELF".into()))
			} else {
				None
			}
		}
	}

	// Custom built-in provider for security domain
	struct SecurityBuiltinsProvider;
	impl BuiltinsProvider for SecurityBuiltinsProvider {
		fn namespace(&self) -> &str {
			"security"
		}

		fn get_builtins(&self) -> BTreeMap<String, hel::BuiltinFn> {
			let mut builtins = BTreeMap::new();

			// security.is_dangerous(format)
			builtins.insert(
				"is_dangerous".to_string(),
				Arc::new(|args: &[Value]| -> Result<Value, hel::EvalError> {
					if args.len() != 1 {
						return Err(hel::EvalError::InvalidOperation(
							"security.is_dangerous expects 1 argument".to_string(),
						));
					}

					match &args[0] {
						Value::String(s) => {
							let is_dangerous = s.as_ref() == "EXE" || s.as_ref() == "DLL";
							Ok(Value::Bool(is_dangerous))
						}
						_ => Ok(Value::Bool(false)),
					}
				}) as hel::BuiltinFn,
			);

			builtins
		}
	}

	let resolver = TestResolver;
	let mut registry = BuiltinsRegistry::new();
	let core = CoreBuiltinsProvider;
	let security = SecurityBuiltinsProvider;
	registry.register(&core).expect("core registration failed");
	registry.register(&security).expect("security registration failed");

	// Test: security.is_dangerous(binary.format) == false (ELF is not dangerous)
	let condition = r#"security.is_dangerous(binary.format) == false"#;
	let result = evaluate_with_context(condition, &resolver, &registry).expect("evaluation failed");
	assert!(result, "ELF format should not be marked as dangerous");

	// Test: security.is_dangerous("EXE") == true
	let condition = r#"security.is_dangerous("EXE") == true"#;
	let result = evaluate_with_context(condition, &resolver, &registry).expect("evaluation failed");
	assert!(result, "EXE format should be marked as dangerous");
}

#[test]
fn test_function_call_in_complex_expression() {
	let resolver = EmptyResolver;
	let mut registry = BuiltinsRegistry::new();
	let provider = CoreBuiltinsProvider;
	registry.register(&provider).expect("registration failed");

	// First test each part individually
	let cond1 = r#"core.len(["a", "b"]) == 2"#;
	let res1 = evaluate_with_context(cond1, &resolver, &registry).expect("eval failed");
	eprintln!("Part 1: {} => {}", cond1, res1);
	assert!(res1, "Part 1 should be true");

	let cond2 = r#"core.contains(["x", "y", "z"], "y") == true"#;
	let res2 = evaluate_with_context(cond2, &resolver, &registry).expect("eval failed");
	eprintln!("Part 2: {} => {}", cond2, res2);
	assert!(res2, "Part 2 should be true");

	// Test complex expression with multiple function calls and operators
	let condition = r#"core.len(["a", "b"]) == 2 AND core.contains(["x", "y", "z"], "y") == true"#;
	let result = evaluate_with_context(condition, &resolver, &registry);
	if let Err(e) = &result {
		eprintln!("Evaluation error: {}", e);
	}
	let result = result.expect("evaluation failed");
	eprintln!("Combined: {} => {}", condition, result);
	assert!(result, "Complex expression with multiple function calls should work");

	// Test with OR
	let condition = r#"core.len(["a"]) == 5 OR core.upper("test") == "TEST""#;
	let result = evaluate_with_context(condition, &resolver, &registry).expect("evaluation failed");
	assert!(result, "OR expression should work with function calls");
}
