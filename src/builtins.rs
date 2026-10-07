//! Pluggable built-in function registry.
//!
//! HEL's core language has no function definitions of its own. Any computation beyond
//! comparison and boolean combination has to come from somewhere, and this module is where
//! that "somewhere" is registered: a [`BuiltinsProvider`] supplies a namespace and a set of
//! functions, and a [`BuiltinsRegistry`] dispatches calls to them.
//!
//! The split exists so a host can add its own vocabulary without forking the parser or the
//! evaluator. This crate ships only [`CoreBuiltinsProvider`], whose operations are generic;
//! anything that knows about a particular domain's data belongs in the host's own provider.
//!
//! ## Namespacing
//!
//! A call in an expression is written `namespace.function(args)`, e.g. `core.len(items)`.
//! Namespaces keep unrelated providers from colliding, and the registry refuses to
//! overwrite one that is already registered (see [`BuiltinsRegistry::register`]).
//!
//! ## Purity
//!
//! A built-in must be pure and deterministic: it sees only its arguments, and must not read the
//! clock, the filesystem or global mutable state. The evaluator calls it from whatever thread
//! happens to be evaluating, so anything else would make results depend on timing.
//!
//! The registry itself is a `BTreeMap`, so iteration order — and therefore
//! [`BuiltinsRegistry::namespaces`] and [`BuiltinsRegistry::functions_in_namespace`] — is
//! stable. Namespaces and the *looked-up* function names are lowercased, so provider key
//! names should be lowercase too.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::{EvalError, Value};

// region:    --- Built-in Function Type

/// The signature every built-in function shares.
///
/// Receives the call's arguments (already evaluated) and returns a [`Value`] or an
/// [`EvalError`]. Implementations must be pure and deterministic — see the module docs —
/// and must validate their own arity, since the registry does not check it.
///
/// The `Arc` exists so a provider's map can be cloned into a registry cheaply; `Send +
/// Sync` is what lets a registry be shared across evaluation threads.
pub type BuiltinFn = Arc<dyn Fn(&[Value]) -> Result<Value, EvalError> + Send + Sync>;

// endregion: --- Built-in Function Type

// region:    --- BuiltinsProvider Trait

/// Supplies a namespace and the built-in functions callable under it.
///
/// Implement this in the host crate for whatever vocabulary that host needs — HEL never
/// needs to know what the functions mean.
///
/// # Examples
///
/// ```
/// use hel::builtins::{BuiltinFn, BuiltinsProvider, BuiltinsRegistry};
/// use hel::{EvalError, Value};
/// use std::collections::BTreeMap;
/// use std::sync::Arc;
///
/// struct MathProvider;
///
/// impl BuiltinsProvider for MathProvider {
///     fn namespace(&self) -> &str {
///         "math"
///     }
///
///     fn get_builtins(&self) -> BTreeMap<String, BuiltinFn> {
///         let mut fns: BTreeMap<String, BuiltinFn> = BTreeMap::new();
///         fns.insert(
///             "double".to_string(),
///             Arc::new(|args: &[Value]| match args {
///                 [Value::Number(n)] => Ok(Value::Number(n * 2.0)),
///                 _ => Err(EvalError::InvalidOperation("math.double expects one number".into())),
///             }),
///         );
///         fns
///     }
/// }
///
/// let mut registry = BuiltinsRegistry::new();
/// registry.register(&MathProvider).expect("registration failed");
/// assert_eq!(
///     registry.call("math", "double", &[Value::Number(21.0)]).expect("call failed"),
///     Value::Number(42.0),
/// );
/// ```
pub trait BuiltinsProvider {
    /// The namespace these functions are registered under, e.g. `"core"`.
    ///
    /// Free-form, but must be unique within a registry: registering a second provider for
    /// a namespace that is already taken is an error.
    fn namespace(&self) -> &str;

    /// The functions this provider exports, keyed by function name.
    ///
    /// Keys should be lowercase, because the registry lowercases names before lookup.
    /// Called once, at registration time; the returned map is moved into the registry.
    fn get_builtins(&self) -> BTreeMap<String, BuiltinFn>;
}

// endregion: --- BuiltinsProvider Trait

// region:    --- BuiltinsRegistry

/// Holds registered providers and dispatches calls to their functions.
///
/// Construct one, [`register`](Self::register) each provider, then hand it to the
/// evaluator — `evaluate_with_context`, `evaluate_with_trace` and friends all take an
/// `Option<&BuiltinsRegistry>`. A registry with no providers is legal but useless: every
/// call fails with [`EvalError::InvalidOperation`].
///
/// Cloning is cheap; the function pointers are shared, not copied.
#[derive(Clone, Default)]
pub struct BuiltinsRegistry {
    /// Namespace -> (function name -> implementation)
    providers: BTreeMap<String, BTreeMap<String, BuiltinFn>>,
}

impl BuiltinsRegistry {
    /// Create a new empty registry.
    ///
    /// # Examples
    ///
    /// ```
    /// use hel::builtins::{BuiltinsRegistry, CoreBuiltinsProvider};
    ///
    /// let mut registry = BuiltinsRegistry::new();
    /// assert!(registry.namespaces().is_empty());
    ///
    /// registry.register(&CoreBuiltinsProvider).expect("registration failed");
    /// assert_eq!(registry.namespaces(), vec!["core".to_string()]);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            providers: BTreeMap::new(),
        }
    }

    /// Register a built-ins provider
    ///
    /// The provider's functions become callable under its [`BuiltinsProvider::namespace`],
    /// which is matched case-insensitively (namespaces and function names are lowercased
    /// on both registration and lookup).
    ///
    /// # Errors
    ///
    /// Returns `Err` naming the namespace if a provider for it is already registered.
    /// Registration never merges or overwrites an existing namespace, and there is no removal
    /// method.
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
    /// `namespace` and `function_name` are matched case-insensitively. The predicate is
    /// pure: it sees only `args`, so the same call always produces the same result.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError::InvalidOperation`] if no provider is registered for
    /// `namespace`, or if the provider has no function of that name. An error returned by
    /// the built-in itself is passed through unchanged.
    pub fn call(
        &self,
        namespace: &str,
        function_name: &str,
        args: &[Value],
    ) -> Result<Value, EvalError> {
        let namespace = namespace.to_lowercase();
        let function_name = function_name.to_lowercase();

        let provider = self.providers.get(&namespace).ok_or_else(|| {
            EvalError::InvalidOperation(format!("Unknown namespace: {}", namespace))
        })?;

        let func = provider.get(&function_name).ok_or_else(|| {
            EvalError::InvalidOperation(format!(
                "Unknown function: {}.{}",
                namespace, function_name
            ))
        })?;

        func(args)
    }

    /// Check whether a function is registered, without calling it.
    ///
    /// Both arguments are matched case-insensitively, exactly as in [`call`](Self::call),
    /// so this is the right way to test a call before making it.
    #[must_use]
    pub fn has_function(&self, namespace: &str, function_name: &str) -> bool {
        let namespace = namespace.to_lowercase();
        let function_name = function_name.to_lowercase();

        self.providers
            .get(&namespace)
            .and_then(|p| p.get(&function_name))
            .is_some()
    }

    /// List every registered namespace, in sorted order.
    #[must_use]
    pub fn namespaces(&self) -> Vec<String> {
        self.providers.keys().cloned().collect()
    }

    /// List every function registered under `namespace`, in sorted order.
    ///
    /// Returns `None` if no provider is registered for that namespace — an empty namespace
    /// and an unknown one are different answers, and this distinguishes them. The namespace
    /// is matched case-insensitively.
    #[must_use]
    pub fn functions_in_namespace(&self, namespace: &str) -> Option<Vec<String>> {
        let namespace = namespace.to_lowercase();
        self.providers
            .get(&namespace)
            .map(|p| p.keys().cloned().collect())
    }
}

// endregion: --- BuiltinsRegistry

// region:    --- Core Built-ins Provider

/// The `core.*` functions: `len`, `contains`, `upper` and `lower`.
///
/// Operations on lists and strings that carry no knowledge of any particular kind of data.
/// Anything that does know about a domain belongs in the host's own [`BuiltinsProvider`], under
/// its own namespace.
///
/// | Call | Behaviour |
/// |---|---|
/// | `core.len(list)` / `core.len(string)` | element count / byte length |
/// | `core.contains(list, value)` | membership by the language's `==` |
/// | `core.contains(string, substring)` | substring test; a non-string needle is `false` |
/// | `core.upper(string)` / `core.lower(string)` | Unicode case conversion |
///
/// Every one of them errors with [`EvalError::InvalidOperation`] on the wrong arity and
/// [`EvalError::TypeMismatch`] on an argument of the wrong type.
///
/// # Examples
///
/// ```
/// use hel::builtins::{BuiltinsRegistry, CoreBuiltinsProvider};
/// use hel::{evaluate_with_context, FactsEvalContext};
///
/// let mut registry = BuiltinsRegistry::new();
/// registry.register(&CoreBuiltinsProvider).expect("registration failed");
///
/// let ctx = FactsEvalContext::new();
/// let matched = evaluate_with_context(
///     r#"core.upper("abc") == "ABC" AND core.len([1, 2, 3]) == 3"#,
///     &ctx,
///     &registry,
/// ).expect("evaluation failed");
/// assert!(matched);
/// ```
pub struct CoreBuiltinsProvider;

impl BuiltinsProvider for CoreBuiltinsProvider {
    fn namespace(&self) -> &str {
        "core"
    }

    fn get_builtins(&self) -> BTreeMap<String, BuiltinFn> {
        let mut builtins = BTreeMap::new();

        // core.len(list)
        builtins.insert(
            "len".to_string(),
            Arc::new(|args: &[Value]| -> Result<Value, EvalError> {
                if args.len() != 1 {
                    return Err(EvalError::InvalidOperation(
                        "core.len expects 1 argument".to_string(),
                    ));
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

        // core.contains(list, value)
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
                        // Membership uses the language's own `==` rather than a second
                        // definition of equality, so `value CONTAINS x` and
                        // `core.contains(value, x)` cannot drift apart.
                        let found = list.iter().any(|item| {
                            crate::compare_new_values(item, &args[1], crate::Comparator::Eq)
                        });
                        Ok(Value::Bool(found))
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

        // core.upper(string)
        builtins.insert(
            "upper".to_string(),
            Arc::new(|args: &[Value]| -> Result<Value, EvalError> {
                if args.len() != 1 {
                    return Err(EvalError::InvalidOperation(
                        "core.upper expects 1 argument".to_string(),
                    ));
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

        // core.lower(string)
        builtins.insert(
            "lower".to_string(),
            Arc::new(|args: &[Value]| -> Result<Value, EvalError> {
                if args.len() != 1 {
                    return Err(EvalError::InvalidOperation(
                        "core.lower expects 1 argument".to_string(),
                    ));
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

        let result = len_fn(&[Value::List(vec![Value::Number(1.0), Value::Number(2.0)])])
            .expect("len failed");
        assert_eq!(result, Value::Number(2.0));

        let result = len_fn(&[Value::String("hello".into())]).expect("len failed");
        assert_eq!(result, Value::Number(5.0));
    }

    #[test]
    fn test_core_contains_builtin() {
        let provider = CoreBuiltinsProvider;
        let builtins = provider.get_builtins();

        let contains_fn = builtins
            .get("contains")
            .expect("contains function not found");

        let list = Value::List(vec![Value::String("a".into()), Value::String("b".into())]);
        let result = contains_fn(&[list, Value::String("a".into())]).expect("contains failed");
        assert_eq!(result, Value::Bool(true));

        let result = contains_fn(&[Value::String("hello".into()), Value::String("ell".into())])
            .expect("contains failed");
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

        let provider = CoreBuiltinsProvider;
        registry.register(&provider).expect("registration failed");

        let result = registry
            .call("core", "len", &[Value::List(vec![Value::Number(1.0)])])
            .expect("call failed");
        assert_eq!(result, Value::Number(1.0));

        let namespaces = registry.namespaces();
        assert_eq!(namespaces, vec!["core"]);

        let functions = registry
            .functions_in_namespace("core")
            .expect("functions not found");
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
                            return Err(EvalError::InvalidOperation(
                                "test.add expects 2 arguments".to_string(),
                            ));
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

        let result = registry
            .call("test", "add", &[Value::Number(1.0), Value::Number(2.0)])
            .expect("call failed");
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
