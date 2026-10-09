//! Minimal host embedding: a resolver, a builtin registry, and `evaluate_with_trace`.
//!
//! Run with `--features acme_provider` to see `hel-template/`'s closed provider registered
//! alongside the core one.

use std::collections::BTreeMap;
use std::error::Error;

use hel::builtins::{BuiltinsRegistry, CoreBuiltinsProvider};
use hel::{HelResolver, Value, evaluate_with_trace};

/// A resolver backed by a map keyed `"object.field"`.
struct InMemoryResolver {
    map: BTreeMap<String, Value>,
}

impl InMemoryResolver {
    fn new() -> Self {
        let mut map = BTreeMap::new();

        map.insert("binary.format".to_string(), Value::String("elf".into()));
        map.insert("security.nx_enabled".to_string(), Value::Bool(true));

        Self { map }
    }
}

impl HelResolver for InMemoryResolver {
    fn resolve_attr(&self, object: &str, field: &str) -> Option<Value> {
        let key = format!("{}.{}", object, field);
        self.map.get(&key).cloned()
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // -- Setup & Fixtures
    let resolver = InMemoryResolver::new();

    // The rule below calls no builtins; registering core shows a host doing it anyway.
    let mut registry = BuiltinsRegistry::new();
    registry
        .register(&CoreBuiltinsProvider)
        .expect("register core builtins");

    // The `acme_provider` feature registers the closed provider template and swaps in a
    // condition that calls it.
    #[cfg(feature = "acme_provider")]
    let condition = {
        let provider = hel_closed_builtins_template::AcmeBuiltins::new();
        registry
            .register(&provider)
            .expect("register acme provider");
        // `acme.score([1, 2, 3])` is the mean, 2.0.
        r#"acme.score([1, 2, 3]) > 1.0 AND binary.format == "elf" AND security.nx_enabled == true"#
    };
    #[cfg(not(feature = "acme_provider"))]
    let condition = r#"binary.format == "elf" AND security.nx_enabled == true"#;

    // -- Exec
    let trace = evaluate_with_trace(condition, &resolver, Some(&registry))?;

    // -- Check
    assert!(trace.result, "expected condition to evaluate to true");
    // Every atom is a `==` on a fact set to the value the literal names.
    assert!(
        trace.atoms.iter().all(|a| a.atom_result),
        "expected every atom to be true"
    );

    println!("{}", trace.pretty_print());

    // `facts_used` is sorted, so it can be compared against a stable order. It lists the
    // attributes the rule read; a builtin call like `acme.score(...)` is not a fact path
    // and does not appear, though attributes passed to a builtin would.
    let expected_facts = vec!["binary.format", "security.nx_enabled"];
    assert_eq!(trace.facts_used(), expected_facts);

    Ok(())
}
