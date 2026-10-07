//! Template for a closed (product-specific) `BuiltinsProvider`.
//!
//! Shows the shape a host crate uses to add its own vocabulary: implement `BuiltinsProvider`,
//! return deterministic `BuiltinFn` closures keyed by function name, and register the provider
//! with a `hel::builtins::BuiltinsRegistry`. Product logic belongs here, not in the public
//! `hel` crate.
//!
//! region:    --- Modules
use std::collections::BTreeMap;
use std::sync::Arc;

use hel::builtins::{BuiltinFn, BuiltinsProvider};
use hel::{EvalError, Value};
// endregion: --- Modules

// region:    --- Provider Definition

/// ACME closed builtins provider (example/template).
///
/// Registered under the `acme` namespace. Built-ins must be deterministic and must not panic;
/// the version is metadata a host can snapshot for audit evidence.
pub struct AcmeBuiltins {
    /// Provider version (semantic or VCS tag)
    pub version: &'static str,
}

impl AcmeBuiltins {
    /// Create a new provider instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            version: ACME_PROVIDER_VERSION,
        }
    }
}

impl Default for AcmeBuiltins {
    fn default() -> Self {
        Self::new()
    }
}

/// Canonical namespace for this provider, as passed to `register`.
pub const ACME_PROVIDER_NAMESPACE: &str = "acme";

/// Version reported in the metadata of every map this provider returns.
pub const ACME_PROVIDER_VERSION: &str = "0.1.0";

// endregion: --- Provider Definition

// region:    --- BuiltinsProvider Implementation

impl BuiltinsProvider for AcmeBuiltins {
    fn namespace(&self) -> &str {
        ACME_PROVIDER_NAMESPACE
    }

    fn get_builtins(&self) -> BTreeMap<String, BuiltinFn> {
        let mut builtins: BTreeMap<String, BuiltinFn> = BTreeMap::new();

        // acme.score(list_of_numbers) -> Number, the mean; errors on non-numbers and bad arity.
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

        // acme.enrich(key, value) -> Map { <key>: value, "provided_by": "acme", "provider_version": <version> }
        builtins.insert(
            "enrich".to_string(),
            Arc::new(|args: &[Value]| -> Result<Value, EvalError> {
                if args.len() != 2 {
                    return Err(EvalError::InvalidOperation(
                        "acme.enrich expects 2 arguments (key:string, value:any)".to_string(),
                    ));
                }

                let key = match &args[0] {
                    Value::String(s) => s.to_string(),
                    _ => {
                        return Err(EvalError::TypeMismatch {
                            expected: "String".to_string(),
                            got: format!("{:?}", args[0]),
                            context: "acme.enrich".to_string(),
                        });
                    }
                };

                // `Value::Map` is keyed by `Arc<str>`, so every key here converts on insert.
                let mut map = BTreeMap::new();
                map.insert(Arc::from(key.as_str()), args[1].clone());
                map.insert(Arc::from("provided_by"), Value::String("acme".into()));
                map.insert(
                    Arc::from("provider_version"),
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

        // -- Exec
        registry.register(&provider).expect("registration failed");

        // acme.score([1.0, 2.0, 3.0]) is the mean, 2.0.
        let args = vec![Value::List(vec![
            Value::Number(1.0),
            Value::Number(2.0),
            Value::Number(3.0),
        ])];

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
                assert_eq!(m.get("foo"), Some(&Value::String("bar".into())));
                assert_eq!(m.get("provided_by"), Some(&Value::String("acme".into())));
                assert_eq!(
                    m.get("provider_version"),
                    Some(&Value::String(ACME_PROVIDER_VERSION.into()))
                );
            }
            _ => panic!("expected map result"),
        }
    }
}
// endregion: --- Tests
