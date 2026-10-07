use std::collections::BTreeMap;
use std::error::Error;

use hel::builtins::{BuiltinsRegistry, CoreBuiltinsProvider};
use hel::{evaluate_with_trace, HelResolver, Value};

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
	let core = CoreBuiltinsProvider;
	registry.register(&core).expect("register core builtins");

	// The `acme_provider` feature attaches the closed provider template and swaps in a condition
	// that calls it. Enabling it also needs a dependency on the template crate, which the example
	// crate's manifest does not carry by default.
	#[cfg(feature = "acme_provider")]
	let condition = {
		let provider = hel_closed_builtins_template::AcmeBuiltins::new();
		registry.register(&provider).expect("register acme provider");
		r#"acme.score([1, 2, 3]) > 2.0 AND binary.format == "elf" AND security.nx_enabled == true"#
	};
	#[cfg(not(feature = "acme_provider"))]
	let condition = r#"binary.format == "elf" AND security.nx_enabled == true"#;

	// -- Exec
	let trace = evaluate_with_trace(&condition, &resolver, Some(&registry))?;

	// -- Check
	assert!(trace.result, "expected condition to evaluate to true");
	assert_eq!(trace.atoms.len(), 2, "expected two atom traces");

	println!("{}", trace.pretty_print());

	// `facts_used` is sorted, so it can be compared against a stable order.
	let facts = trace.facts_used();
	println!("{}", trace.pretty_print());
	assert_eq!(
		facts,
		vec!["binary.format".to_string(), "security.nx_enabled".to_string()]
	);

	Ok(())
}
