// Template closed BuiltinsProvider implementation for HEL.
//
// This is an example skeleton showing how a product can provide a closed
// builtins provider and register deterministic, versioned functions with the
// `hel::builtins::BuiltinsRegistry`.
//
// The code is intentionally small and focuses on:
// - implementing `BuiltinsProvider`
// - providing deterministic builtins as `BuiltinFn` closures
// - exposing provider metadata (via constants in this template)
// - a small test that demonstrates registration and invocation
//
// Note: This crate is a template and should live in a closed/product workspace.
// Keep product-specific logic out of the public `hel` crate and inject it via
// a provider like this one.
//
// region:    --- Modules
use std::collections::BTreeMap;
use std::sync::Arc;

use hel::builtins::{BuiltinFn, BuiltinsProvider};
use hel::{EvalError, Value};
// endregion: --- Modules

// region:    --- Provider Definition

/// ACME closed builtins provider (example/template)
///
/// - Namespace: `acme`
/// - Provider version: `0.1.0`
///
/// Implementations SHOULD:
/// - Be deterministic (same input -> same output)
/// - Return `Result<Value, EvalError>` and never panic
/// - Emit metadata (version) that hosts can snapshot for audit evidence
pub struct AcmeBuiltins {
	/// Provider version (semantic or VCS tag)
	pub version: &'static str,
}

impl AcmeBuiltins {
	/// Create a new provider instance
	pub fn new() -> Self {
		Self {
			version: ACME_PROVIDER_VERSION,
		}
	}
}

/// Canonical metadata for this provider (useful for registry snapshots)
pub const ACME_PROVIDER_NAMESPACE: &str = "acme";
pub const ACME_PROVIDER_VERSION: &str = "0.1.0";

// endregion: --- Provider Definition

// region:    --- BuiltinsProvider Implementation

impl BuiltinsProvider for AcmeBuiltins {
	fn namespace(&self) -> &str {
		ACME_PROVIDER_NAMESPACE
	}

	fn get_builtins(&self) -> BTreeMap<String, BuiltinFn> {
		let mut builtins: BTreeMap<String, BuiltinFn> = BTreeMap::new();

		// acme.score(list_of_numbers) -> Number (average)
		// Deterministic: average of input numbers. Errors on wrong types / arity.
		builtins.insert(
			"score".to_string(),
			Arc::new(|args: &[Value]| -> Result<Value, EvalError> {
				if args.len() != 1 {
					return Err(EvalError::InvalidOperation(
						"acme.score expects 1 argument (list of numbers)".to_string(),
					));
				}

				match &args[0] {
					Value::List(items) => {
						if items.is_empty() {
							return Ok(Value::Number(0.0));
						}
						let mut sum = 0.0f64;
						let mut count = 0usize;
						for item in items {
							match item {
								Value::Number(n) => {
									sum += *n;
									count += 1;
								}
								_ => {
									return Err(EvalError::TypeMismatch {
										expected: "Number".to_string(),
										got: format!("{:?}", item),
										context: "acme.score".to_string(),
									});
								}
							}
						}
						Ok(Value::Number(sum / (count as f64)))
					}
					_ => Err(EvalError::TypeMismatch {
						expected: "List".to_string(),
						got: format!("{:?}", args[0]),
						context: "acme.score".to_string(),
					}),
				}
			}) as BuiltinFn,
		);

		// acme.enrich(key, value) -> Map { "key": value, "provided_by": "acme", "prov_ver": <version> }
		// Simple deterministic enrichment that returns a small map.
		builtins.insert(
			"enrich".to_string(),
			Arc::new(|args: &[Value]| -> Result<Value, EvalError> {
				if args.len() != 2 {
					return Err(EvalError::InvalidOperation(
						"acme.enrich expects 2 arguments (key:string, value:any)".to_string(),
					));
				}

				// key must be a string
				let key = match &args[0] {
					Value::String(s) => s.to_string(),
					_ => {
						return Err(EvalError::TypeMismatch {
							expected: "String".to_string(),
							got: format!("{:?}", args[0]),
							context: "acme.enrich".to_string(),
						})
					}
				};

				let mut map = std::collections::BTreeMap::new();
				// Insert the original pair under provided key
				map.insert(key.clone(), args[1].clone());
				// Add provider metadata (deterministic)
				map.insert("provided_by".to_string(), Value::String("acme".into()));
				map.insert(
					"provider_version".to_string(),
					Value::String(ACME_PROVIDER_VERSION.into()),
				);

				Ok(Value::Map(map))
			}) as BuiltinFn,
		);

		builtins
	}
}
// endregion: --- BuiltinsProvider Implementation

// region:    --- Tests

#[cfg(test)]
mod tests {
	use super::*;
	use hel::builtins::BuiltinsRegistry;

	#[test]
	fn test_acme_provider_register_and_score() {
		// -- Setup & Fixtures
		let provider = AcmeBuiltins::new();
		let mut registry = BuiltinsRegistry::new();

		// -- Exec: register provider and call builtins
		registry.register(&provider).expect("registration failed");

		// Prepare args for acme.score([1.0, 2.0, 3.0]) -> avg = 2.0
		let args = vec![Value::List(vec![Value::Number(1.0), Value::Number(2.0), Value::Number(3.0)])];

		let result = registry.call("acme", "score", &args).expect("call failed");
		// -- Check
		assert_eq!(result, Value::Number(2.0));
	}

	#[test]
	fn test_acme_enrich_map_shape() {
		// -- Setup & Fixtures
		let provider = AcmeBuiltins::new();
		let mut registry = BuiltinsRegistry::new();
		registry.register(&provider).expect("registration failed");

		// -- Exec
		let key = Value::String("foo".into());
		let val = Value::String("bar".into());
		let result = registry
			.call("acme", "enrich", &[key.clone(), val.clone()])
			.expect("enrich failed");

		// -- Check
		match result {
			Value::Map(m) => {
				// original key present
				assert!(m.contains_key("foo"));
				// provider metadata present
				assert_eq!(m.get("provided_by"), Some(&Value::String("acme".into())));
				assert_eq!(
					m.get("provider_version"),
					Some(&Value::String(ACME_PROVIDER_VERSION.into()))
				);
				// value preserved
				assert_eq!(m.get("foo"), Some(&Value::String("bar".into())));
			}
			_ => panic!("expected map result"),
		}
	}
}
// endregion: --- Tests
