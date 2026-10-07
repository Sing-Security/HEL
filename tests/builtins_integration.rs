//! Built-in functions driven end-to-end through the evaluator.

use hel::{
    evaluate_with_context, BuiltinsProvider, BuiltinsRegistry, CoreBuiltinsProvider, HelResolver,
    Value,
};
use std::collections::BTreeMap;
use std::sync::Arc;

// Answers nothing — these tests use only literals and function calls.
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

    let condition = r#"core.contains(["a", "b", "c"], "b") == true"#;
    let result = evaluate_with_context(condition, &resolver, &registry).expect("evaluation failed");
    assert!(result, "core.contains should find 'b' in list");

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

    let condition = r#"core.upper("hello") == "HELLO""#;
    let result = evaluate_with_context(condition, &resolver, &registry).expect("evaluation failed");
    assert!(result, "core.upper should convert to uppercase");

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
    registry
        .register(&security)
        .expect("security registration failed");

    let condition = r#"security.is_dangerous(binary.format) == false"#;
    let result = evaluate_with_context(condition, &resolver, &registry).expect("evaluation failed");
    assert!(result, "ELF format should not be marked as dangerous");

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

    let condition = r#"core.len(["a", "b"]) == 2 AND core.contains(["x", "y", "z"], "y") == true"#;
    let result = evaluate_with_context(condition, &resolver, &registry).expect("evaluation failed");
    assert!(result, "AND of two function calls should be true");

    // One false operand is enough to fail the AND.
    let condition = r#"core.len(["a", "b"]) == 2 AND core.contains(["x"], "y") == true"#;
    let result = evaluate_with_context(condition, &resolver, &registry).expect("evaluation failed");
    assert!(!result, "AND with a false call should be false");

    let condition = r#"core.len(["a"]) == 5 OR core.upper("test") == "TEST""#;
    let result = evaluate_with_context(condition, &resolver, &registry).expect("evaluation failed");
    assert!(result, "OR expression should work with function calls");
}

#[test]
fn test_core_contains_agrees_with_contains_operator() {
    let resolver = EmptyResolver;
    let mut registry = BuiltinsRegistry::new();
    let provider = CoreBuiltinsProvider;
    registry.register(&provider).expect("registration failed");

    // `core.contains` and the language's `CONTAINS` must answer the same question the
    // same way; they are two spellings of one operation, not two definitions of it.
    for element in ["a", "b", "c"] {
        let operator_form = format!(r#"["a", "b", "c"] CONTAINS "{}""#, element);
        let builtin_form = format!(r#"core.contains(["a", "b", "c"], "{}") == true"#, element);

        let via_operator = evaluate_with_context(&operator_form, &resolver, &registry)
            .expect("operator form failed");
        let via_builtin = evaluate_with_context(&builtin_form, &resolver, &registry)
            .expect("builtin form failed");
        assert_eq!(via_operator, via_builtin, "disagreement for {}", element);
        assert!(via_operator);
    }

    // And they agree when the answer is "no".
    let via_operator = evaluate_with_context(r#"["a"] CONTAINS "z""#, &resolver, &registry)
        .expect("operator form failed");
    let via_builtin =
        evaluate_with_context(r#"core.contains(["a"], "z") == true"#, &resolver, &registry)
            .expect("builtin form failed");
    assert_eq!(via_operator, via_builtin);
    assert!(!via_operator);
}
