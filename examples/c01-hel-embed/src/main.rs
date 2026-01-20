use std::collections::BTreeMap;
use std::error::Error;

use hel::builtins::{BuiltinsRegistry, CoreBuiltinsProvider};
use hel::{evaluate_with_trace, HelResolver, Value};

/// Simple in-memory resolver for examples.
///
/// It stores values keyed by `"object.field"` (e.g., "binary.format") and returns
/// `Some(Value)` for found attributes or `None` when missing. This keeps the
/// example compact while exercising the public `HelResolver` contract.
struct InMemoryResolver {
	map: BTreeMap<String, Value>,
}

impl InMemoryResolver {
	fn new() -> Self {
		let mut map = BTreeMap::new();

		// Populate a couple of sample facts used by the example rule.
		map.insert("binary.format".to_string(), Value::String("elf".into()));
		map.insert("security.nx_enabled".to_string(), Value::Bool(true));

		// Example list/map facts could also be added if you want to exercise builtins.
		// e.g. map.insert("files".to_string(), Value::List(vec![ ... ]));

		Self { map }
	}
}

impl HelResolver for InMemoryResolver {
	/// Resolve an attribute path like `binary.format` into a HEL `Value`.
	/// Returns `None` for missing attributes (interpreted as `null` by HEL).
	fn resolve_attr(&self, object: &str, field: &str) -> Option<Value> {
		let key = format!("{}.{}", object, field);
		self.map.get(&key).cloned()
	}
}

fn main() -> Result<(), Box<dyn Error>> {
	// -- Setup & Fixtures
	let resolver = InMemoryResolver::new();

	// Build a builtin registry and register the core (open) builtins.
	// In this example the rule does not call builtins, but we show how a host
	// would attach core builtins and additional closed providers.
	let mut registry = BuiltinsRegistry::new();
	let core = CoreBuiltinsProvider;
	registry.register(&core).expect("register core builtins");

	// Example HEL condition (mirrors tests in the crate)
	// Default condition (no closed provider)
	let mut condition = r#"binary.format == "elf" AND security.nx_enabled == true"#.to_string();

	// Optionally register the closed-provider template and extend the condition to
	// call a builtin (`acme.score`) so the trace shows a resolved builtin call.
	//
	// To enable this path compile the example with the `acme_provider` feature
	// and add a workspace/local dependency on the provider crate:
	// `hel_closed_builtins_template = { path = "../../../products/hel_closed_builtins_template", optional = true }`
	#[cfg(feature = "acme_provider")]
	{
		// The provider crate is expected to expose `AcmeBuiltins`.
		let provider = hel_closed_builtins_template::AcmeBuiltins::new();
		registry.register(&provider).expect("register acme provider");

		// Extend the condition to include a builtin call:
		condition =
			r#"acme.score([1, 2, 3]) > 2.0 AND binary.format == "elf" AND security.nx_enabled == true"#.to_string();
	}

	// -- Exec
	let trace = evaluate_with_trace(condition, &resolver, Some(&registry))?;

	// -- Check / Inspect results (use crate-provided deterministic Display)
	println!("{}", trace);

	// Basic assertions for the example (will panic if violated; tests/examples typically assert)
	assert!(trace.result, "expected condition to evaluate to true");
	assert_eq!(trace.atoms.len(), 2, "expected two atom traces");

	// Demonstrate deterministic facts_used ordering and print the crate-provided pretty trace
	let facts = trace.facts_used();
	println!("{}", trace.pretty_print());
	assert_eq!(
		facts,
		vec!["binary.format".to_string(), "security.nx_enabled".to_string()]
	);

	Ok(())
}
