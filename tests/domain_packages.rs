//! Integration tests for HEL domain packages
//!
//! These tests are self-contained: they create temporary packages on disk,
//! add the temp directory as a search path, and then load and validate them.

use hel::PackageRegistry;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

fn write_package(dir: &Path, name: &str, version: &str, schemas: &[(&str, &str)]) {
    // Package directory layout:
    // <dir>/<name>/hel-package.toml
    // <dir>/<name>/<schema_path...>
    let pkg_dir = dir.join(name);
    fs::create_dir_all(&pkg_dir).expect("failed to create package directory");

    // Ensure schema subdirectories exist
    for (rel_path, _) in schemas {
        if let Some(parent) = Path::new(rel_path).parent() {
            fs::create_dir_all(pkg_dir.join(parent)).expect("failed to create schema parent dirs");
        }
    }

    // Manifest
    let schema_paths: Vec<String> = schemas.iter().map(|(p, _)| p.to_string()).collect();
    let manifest = format!(
        "\
name = \"{name}\"
version = \"{version}\"
schemas = [{schemas}]
",
        name = name,
        version = version,
        schemas = schema_paths
            .iter()
            .map(|p| format!("\"{}\"", p))
            .collect::<Vec<_>>()
            .join(", ")
    );

    fs::write(pkg_dir.join("hel-package.toml"), manifest).expect("failed to write manifest");

    // Schemas
    for (rel_path, content) in schemas {
        fs::write(pkg_dir.join(rel_path), content).expect("failed to write schema file");
    }
}

fn create_test_domains_dir() -> (TempDir, PathBuf) {
    let temp = TempDir::new().expect("failed to create temp dir");
    let root = temp.path().to_path_buf();

    // Minimal "security-binary" package with the types these tests assert on.
    write_package(
        &root,
        "security-binary",
        "0.1.0",
        &[(
            "schema/00_domain.hel",
            r#"
type Binary {
    arch: String
}

type Security {
    nx: Bool
}

type Section {
    name: String
}

type Import {
    name: String
}

type TaintFlow {
    source: String
    sink: String
}
"#,
        )],
    );

    // Minimal "sales-crm" package with the types these tests assert on.
    write_package(
        &root,
        "sales-crm",
        "0.1.0",
        &[(
            "schema/00_domain.hel",
            r#"
type Lead {
    id: String
}

type Contact {
    email: String
}

type Enrichment {
    provider: String
}
"#,
        )],
    );

    (temp, root)
}

#[test]
fn test_load_security_binary_package() {
    let (_temp, domains_dir) = create_test_domains_dir();

    let mut registry = PackageRegistry::new();
    registry.add_search_path(domains_dir);

    let package = registry
        .load_package("security-binary")
        .expect("Failed to load security-binary package");

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
    let (_temp, domains_dir) = create_test_domains_dir();

    let mut registry = PackageRegistry::new();
    registry.add_search_path(domains_dir);

    let package = registry
        .load_package("sales-crm")
        .expect("Failed to load sales-crm package");

    assert_eq!(package.manifest.name, "sales-crm");
    assert_eq!(package.manifest.version, "0.1.0");

    // Check that types are loaded
    assert!(package.schema.get_type("Lead").is_some());
    assert!(package.schema.get_type("Contact").is_some());
    assert!(package.schema.get_type("Enrichment").is_some());
}

#[test]
fn test_build_type_environment_with_multiple_packages() {
    let (_temp, domains_dir) = create_test_domains_dir();

    let mut registry = PackageRegistry::new();
    registry.add_search_path(domains_dir);

    // Load both packages
    registry
        .load_package("security-binary")
        .expect("Failed to load security-binary");
    registry
        .load_package("sales-crm")
        .expect("Failed to load sales-crm");

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
    let (_temp, domains_dir) = create_test_domains_dir();

    let mut registry = PackageRegistry::new();
    registry.add_search_path(domains_dir);

    // Load packages
    registry
        .load_package("security-binary")
        .expect("Failed to load");
    registry.load_package("sales-crm").expect("Failed to load");

    // Get packages after loading
    let sec = registry
        .get_package("security-binary")
        .expect("Package not found");
    let sales = registry
        .get_package("sales-crm")
        .expect("Package not found");

    // Namespaces should match package names
    assert_eq!(sec.namespace(), "security-binary");
    assert_eq!(sales.namespace(), "sales-crm");

    // Build environment and ensure no collisions
    let env = registry
        .build_type_environment(&["security-binary".to_string(), "sales-crm".to_string()])
        .expect("Failed to build environment");

    // All types should be qualified
    let type_count = env.types.len();
    assert!(type_count > 5, "Expected multiple types from both packages");
}
