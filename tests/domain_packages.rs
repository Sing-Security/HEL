//! Integration tests for HEL domain packages
//!
//! These tests demonstrate loading domain packages from the domains/ directory.

use hel::PackageRegistry;
use std::path::PathBuf;

fn get_domains_path() -> PathBuf {
	// Test runs from workspace root, domains/ is at repo root
	PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.parent()
		.unwrap()
		.parent()
		.unwrap()
		.parent()
		.unwrap()
		.join("domains")
}

#[test]
fn test_load_security_binary_package() {
	let mut registry = PackageRegistry::new();
	registry.add_search_path(get_domains_path());

	let package = registry.load_package("security-binary").expect("Failed to load security-binary package");

	assert_eq!(package.manifest.name, "security-binary");
	assert_eq!(package.manifest.version, "0.1.0");

	// Check that types are loaded
	assert!(package.schema.get_type("Binary").is_some());
	assert!(package.schema.get_type("Security").is_some());
	assert!(package.schema.get_type("Section").is_some());
	assert!(package.schema.get_type("Import").is_some());
	assert!(package.schema.get_type("TaintFlow").is_some());
}

#[test]
fn test_load_sales_crm_package() {
	let mut registry = PackageRegistry::new();
	registry.add_search_path(get_domains_path());

	let package = registry.load_package("sales-crm").expect("Failed to load sales-crm package");

	assert_eq!(package.manifest.name, "sales-crm");
	assert_eq!(package.manifest.version, "0.1.0");

	// Check that types are loaded
	assert!(package.schema.get_type("Lead").is_some());
	assert!(package.schema.get_type("Contact").is_some());
	assert!(package.schema.get_type("Enrichment").is_some());
}

#[test]
fn test_build_type_environment_with_multiple_packages() {
	let mut registry = PackageRegistry::new();
	registry.add_search_path(get_domains_path());

	// Load both packages
	registry.load_package("security-binary").expect("Failed to load security-binary");
	registry.load_package("sales-crm").expect("Failed to load sales-crm");

	// Build type environment
	let env = registry
		.build_type_environment(&["security-binary".to_string(), "sales-crm".to_string()])
		.expect("Failed to build type environment");

	// Check qualified type names
	assert!(env.get_type("security-binary.Binary").is_some());
	assert!(env.get_type("security-binary.Section").is_some());
	assert!(env.get_type("sales-crm.Lead").is_some());
	assert!(env.get_type("sales-crm.Contact").is_some());

	// Note: Cross-package validation would require qualified type references in schemas
	// For now, we just check that types are loaded correctly
}

#[test]
fn test_package_namespace_separation() {
	let mut registry = PackageRegistry::new();
	registry.add_search_path(get_domains_path());

	// Load packages
	registry.load_package("security-binary").expect("Failed to load");
	registry.load_package("sales-crm").expect("Failed to load");

	// Get packages after loading
	let sec = registry.get_package("security-binary").expect("Package not found");
	let sales = registry.get_package("sales-crm").expect("Package not found");

	// Namespaces should match package names
	assert_eq!(sec.namespace(), "security-binary");
	assert_eq!(sales.namespace(), "sales-crm");

	// Build environment and ensure no collisions
	let env = registry
		.build_type_environment(&["security-binary".to_string(), "sales-crm".to_string()])
		.expect("Failed to build environment");

	// All types should be qualified
	let type_count = env.types.len();
	assert!(type_count > 10, "Expected multiple types from both packages");
}
