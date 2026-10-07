//! Package loading and resolution over real directories on disk: each test writes its own
//! packages into a temp dir, adds it as a search path, and loads from there.

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

    assert!(package.schema.get_type("Lead").is_some());
    assert!(package.schema.get_type("Contact").is_some());
    assert!(package.schema.get_type("Enrichment").is_some());
}

#[test]
fn test_build_type_environment_with_multiple_packages() {
    let (_temp, domains_dir) = create_test_domains_dir();

    let mut registry = PackageRegistry::new();
    registry.add_search_path(domains_dir);

    registry
        .load_package("security-binary")
        .expect("Failed to load security-binary");
    registry
        .load_package("sales-crm")
        .expect("Failed to load sales-crm");

    let env = registry
        .build_type_environment(&["security-binary".to_string(), "sales-crm".to_string()])
        .expect("Failed to build type environment");

    assert!(env.get_type("security-binary.Binary").is_some());
    assert!(env.get_type("security-binary.Section").is_some());
    assert!(env.get_type("sales-crm.Lead").is_some());
    assert!(env.get_type("sales-crm.Contact").is_some());

    // Cross-package validation would need qualified type references in schemas, which these
    // fixtures do not carry, so this test stops at the types having loaded at all.
}

#[test]
fn test_package_namespace_separation() {
    let (_temp, domains_dir) = create_test_domains_dir();

    let mut registry = PackageRegistry::new();
    registry.add_search_path(domains_dir);

    registry
        .load_package("security-binary")
        .expect("Failed to load");
    registry.load_package("sales-crm").expect("Failed to load");

    let sec = registry
        .get_package("security-binary")
        .expect("Package not found");
    let sales = registry
        .get_package("sales-crm")
        .expect("Package not found");

    assert_eq!(sec.namespace(), "security-binary");
    assert_eq!(sales.namespace(), "sales-crm");

    let env = registry
        .build_type_environment(&["security-binary".to_string(), "sales-crm".to_string()])
        .expect("Failed to build environment");

    // Both packages' types are present under their qualified names.
    let type_count = env.types.len();
    assert!(type_count > 5, "Expected multiple types from both packages");
}
