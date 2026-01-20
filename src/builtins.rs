//! Built-in function registry for HEL
//!
//! This module provides a pluggable system for domain-specific built-in functions.
//! Domains can register deterministic functions under their namespace without
//! modifying the HEL core language.
//!
//! ## Architecture
//! - BuiltinsProvider trait: defines how to provide built-in functions
//! - BuiltinsRegistry: namespace-aware function dispatcher
//! - Built-ins are pure and deterministic (no I/O, no global state)
//!
//! ## Namespacing
//! - Functions are called as `namespace.function_name(args)`
//! - Example: `security.contains(list, value)` or `sales.email_is_valid(email)`
//!
//! ## Determinism
//! - All built-ins must be pure functions
//! - Registry uses BTreeMap for stable iteration order
//! - Function names are normalized to lowercase for consistency

use std::collections::BTreeMap;
use std::sync::Arc;

use super::{EvalError, Value};

// region:    --- Built-in Function Type

/// A built-in function signature
///
/// Takes a list of arguments and returns a Value or error.
/// Must be deterministic and pure (no I/O, no global state).
pub type BuiltinFn = Arc<dyn Fn(&[Value]) -> Result<Value, EvalError> + Send + Sync>;

// endregion: --- Built-in Function Type

// region:    --- BuiltinsProvider Trait

/// Trait for providing built-in functions for a domain
///
/// Domains implement this trait to provide their custom functions.
/// The namespace is typically the domain package name.
pub trait BuiltinsProvider {
	/// Get the namespace for these built-ins (e.g., "security", "sales")
	fn namespace(&self) -> &str;

	/// Get all built-in functions provided by this domain
	///
	/// Returns a map of function name (lowercase) -> implementation
	fn get_builtins(&self) -> BTreeMap<String, BuiltinFn>;
}

// endregion: --- BuiltinsProvider Trait

// region:    --- BuiltinsRegistry

/// Registry for namespace-aware built-in functions
///
/// Manages multiple providers and dispatches function calls deterministically.
#[derive(Clone)]
pub struct BuiltinsRegistry {
	/// Namespace -> (function_name -> implementation)
	providers: BTreeMap<String, BTreeMap<String, BuiltinFn>>,
}

impl BuiltinsRegistry {
	/// Create a new empty registry
	pub fn new() -> Self {
		Self {
			providers: BTreeMap::new(),
		}
	}

	/// Register a built-ins provider
	///
	/// Returns error if the namespace is already registered
	pub fn register(&mut self, provider: &dyn BuiltinsProvider) -> Result<(), String> {
		let namespace = provider.namespace().to_lowercase();

		if self.providers.contains_key(&namespace) {
			return Err(format!("Namespace '{}' is already registered", namespace));
		}

		let builtins = provider.get_builtins();
		self.providers.insert(namespace, builtins);

		Ok(())
	}

	/// Call a built-in function by qualified name
	///
	/// # Arguments
	/// * `namespace` - The namespace (e.g., "security")
	/// * `function_name` - The function name (e.g., "contains")
	/// * `args` - The function arguments
	///
	/// # Returns
	/// The function result, or error if function not found or execution fails
	pub fn call(&self, namespace: &str, function_name: &str, args: &[Value]) -> Result<Value, EvalError> {
		let namespace = namespace.to_lowercase();
		let function_name = function_name.to_lowercase();

		let provider = self.providers.get(&namespace).ok_or_else(|| EvalError::InvalidOperation(format!("Unknown namespace: {}", namespace)))?;

		let func = provider
			.get(&function_name)
			.ok_or_else(|| EvalError::InvalidOperation(format!("Unknown function: {}.{}", namespace, function_name)))?;

		func(args)
	}

	/// Check if a function exists
	pub fn has_function(&self, namespace: &str, function_name: &str) -> bool {
		let namespace = namespace.to_lowercase();
		let function_name = function_name.to_lowercase();

		self.providers
			.get(&namespace)
			.and_then(|p| p.get(&function_name))
			.is_some()
	}

	/// List all registered namespaces
	pub fn namespaces(&self) -> Vec<String> {
		self.providers.keys().cloned().collect()
	}

	/// List all functions in a namespace
	pub fn functions_in_namespace(&self, namespace: &str) -> Option<Vec<String>> {
		let namespace = namespace.to_lowercase();
		self.providers.get(&namespace).map(|p| p.keys().cloned().collect())
	}
}

impl Default for BuiltinsRegistry {
	fn default() -> Self {
		Self::new()
	}
}

// endregion: --- BuiltinsRegistry

// region:    --- Core Built-ins Provider (Open Implementation)

/// Core built-ins provider for common/open functions
///
/// These are generic, product-agnostic functions that are safe to open-source.
pub struct CoreBuiltinsProvider;

impl BuiltinsProvider for CoreBuiltinsProvider {
	fn namespace(&self) -> &str {
		"core"
	}

	fn get_builtins(&self) -> BTreeMap<String, BuiltinFn> {
		let mut builtins = BTreeMap::new();

		// core.len(list) - get length of list
		builtins.insert(
			"len".to_string(),
			Arc::new(|args: &[Value]| -> Result<Value, EvalError> {
				if args.len() != 1 {
					return Err(EvalError::InvalidOperation("core.len expects 1 argument".to_string()));
				}

				match &args[0] {
					Value::List(list) => Ok(Value::Number(list.len() as f64)),
					Value::String(s) => Ok(Value::Number(s.len() as f64)),
					_ => Err(EvalError::TypeMismatch {
						expected: "List or String".to_string(),
						got: format!("{:?}", args[0]),
						context: "core.len".to_string(),
					}),
				}
			}) as BuiltinFn,
		);

		// core.contains(list, value) - check if list contains value
		builtins.insert(
			"contains".to_string(),
			Arc::new(|args: &[Value]| -> Result<Value, EvalError> {
				if args.len() != 2 {
					return Err(EvalError::InvalidOperation(
						"core.contains expects 2 arguments".to_string(),
					));
				}

				match &args[0] {
					Value::List(list) => {
						let result = list.iter().any(|item| values_equal(item, &args[1]));
						Ok(Value::Bool(result))
					}
					Value::String(haystack) => match &args[1] {
						Value::String(needle) => Ok(Value::Bool(haystack.contains(&**needle))),
						_ => Ok(Value::Bool(false)),
					},
					_ => Err(EvalError::TypeMismatch {
						expected: "List or String".to_string(),
						got: format!("{:?}", args[0]),
						context: "core.contains".to_string(),
					}),
				}
			}) as BuiltinFn,
		);

		// core.upper(string) - convert to uppercase
		builtins.insert(
			"upper".to_string(),
			Arc::new(|args: &[Value]| -> Result<Value, EvalError> {
				if args.len() != 1 {
					return Err(EvalError::InvalidOperation("core.upper expects 1 argument".to_string()));
				}

				match &args[0] {
					Value::String(s) => Ok(Value::String(s.to_uppercase().into())),
					_ => Err(EvalError::TypeMismatch {
						expected: "String".to_string(),
						got: format!("{:?}", args[0]),
						context: "core.upper".to_string(),
					}),
				}
			}) as BuiltinFn,
		);

		// core.lower(string) - convert to lowercase
		builtins.insert(
			"lower".to_string(),
			Arc::new(|args: &[Value]| -> Result<Value, EvalError> {
				if args.len() != 1 {
					return Err(EvalError::InvalidOperation("core.lower expects 1 argument".to_string()));
				}

				match &args[0] {
					Value::String(s) => Ok(Value::String(s.to_lowercase().into())),
					_ => Err(EvalError::TypeMismatch {
						expected: "String".to_string(),
						got: format!("{:?}", args[0]),
						context: "core.lower".to_string(),
					}),
				}
			}) as BuiltinFn,
		);

		builtins
	}
}

/// Helper function to compare values for equality
fn values_equal(a: &Value, b: &Value) -> bool {
	match (a, b) {
		(Value::Null, Value::Null) => true,
		(Value::Bool(a), Value::Bool(b)) => a == b,
		(Value::String(a), Value::String(b)) => a == b,
		(Value::Number(a), Value::Number(b)) => a == b,
		(Value::List(a), Value::List(b)) => {
			a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| values_equal(x, y))
		}
		_ => false,
	}
}

// endregion: --- Core Built-ins Provider (Open Implementation)

// region:    --- Tests

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_core_len_builtin() {
		let provider = CoreBuiltinsProvider;
		let builtins = provider.get_builtins();

		let len_fn = builtins.get("len").expect("len function not found");

		// Test with list
		let result = len_fn(&[Value::List(vec![Value::Number(1.0), Value::Number(2.0)])]).expect("len failed");
		assert_eq!(result, Value::Number(2.0));

		// Test with string
		let result = len_fn(&[Value::String("hello".into())]).expect("len failed");
		assert_eq!(result, Value::Number(5.0));
	}

	#[test]
	fn test_core_contains_builtin() {
		let provider = CoreBuiltinsProvider;
		let builtins = provider.get_builtins();

		let contains_fn = builtins.get("contains").expect("contains function not found");

		// Test list contains
		let list = Value::List(vec![Value::String("a".into()), Value::String("b".into())]);
		let result = contains_fn(&[list, Value::String("a".into())]).expect("contains failed");
		assert_eq!(result, Value::Bool(true));

		// Test string contains
		let result = contains_fn(&[Value::String("hello".into()), Value::String("ell".into())]).expect("contains failed");
		assert_eq!(result, Value::Bool(true));
	}

	#[test]
	fn test_core_upper_lower() {
		let provider = CoreBuiltinsProvider;
		let builtins = provider.get_builtins();

		let upper_fn = builtins.get("upper").expect("upper not found");
		let lower_fn = builtins.get("lower").expect("lower not found");

		let result = upper_fn(&[Value::String("hello".into())]).expect("upper failed");
		assert_eq!(result, Value::String("HELLO".into()));

		let result = lower_fn(&[Value::String("WORLD".into())]).expect("lower failed");
		assert_eq!(result, Value::String("world".into()));
	}

	#[test]
	fn test_builtins_registry() {
		let mut registry = BuiltinsRegistry::new();

		// Register core provider
		let provider = CoreBuiltinsProvider;
		registry.register(&provider).expect("registration failed");

		// Test function call
		let result = registry
			.call("core", "len", &[Value::List(vec![Value::Number(1.0)])])
			.expect("call failed");
		assert_eq!(result, Value::Number(1.0));

		// Test namespace listing
		let namespaces = registry.namespaces();
		assert_eq!(namespaces, vec!["core"]);

		// Test function listing
		let functions = registry.functions_in_namespace("core").expect("functions not found");
		assert!(functions.contains(&"len".to_string()));
		assert!(functions.contains(&"contains".to_string()));
	}

	#[test]
	fn test_custom_builtin_provider() {
		struct TestProvider;

		impl BuiltinsProvider for TestProvider {
			fn namespace(&self) -> &str {
				"test"
			}

			fn get_builtins(&self) -> BTreeMap<String, BuiltinFn> {
				let mut builtins = BTreeMap::new();

				// test.add(a, b)
				builtins.insert(
					"add".to_string(),
					Arc::new(|args: &[Value]| -> Result<Value, EvalError> {
						if args.len() != 2 {
							return Err(EvalError::InvalidOperation("test.add expects 2 arguments".to_string()));
						}

						match (&args[0], &args[1]) {
							(Value::Number(a), Value::Number(b)) => Ok(Value::Number(a + b)),
							_ => Err(EvalError::TypeMismatch {
								expected: "Number".to_string(),
								got: "other".to_string(),
								context: "test.add".to_string(),
							}),
						}
					}) as BuiltinFn,
				);

				builtins
			}
		}

		let mut registry = BuiltinsRegistry::new();
		let provider = TestProvider;
		registry.register(&provider).expect("registration failed");

		let result = registry.call("test", "add", &[Value::Number(1.0), Value::Number(2.0)]).expect("call failed");
		assert_eq!(result, Value::Number(3.0));
	}

	#[test]
	fn test_namespace_collision() {
		struct Provider1;
		impl BuiltinsProvider for Provider1 {
			fn namespace(&self) -> &str {
				"test"
			}
			fn get_builtins(&self) -> BTreeMap<String, BuiltinFn> {
				BTreeMap::new()
			}
		}

		struct Provider2;
		impl BuiltinsProvider for Provider2 {
			fn namespace(&self) -> &str {
				"test"
			}
			fn get_builtins(&self) -> BTreeMap<String, BuiltinFn> {
				BTreeMap::new()
			}
		}

		let mut registry = BuiltinsRegistry::new();
		let p1 = Provider1;
		let p2 = Provider2;

		registry.register(&p1).expect("first registration failed");
		let result = registry.register(&p2);
		assert!(result.is_err());
		assert!(result.unwrap_err().contains("already registered"));
	}
}

// endregion: --- Tests
