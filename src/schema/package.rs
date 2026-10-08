//! Packages of HEL domain schemas.
//!
//! A package is a directory with a `hel-package.toml` manifest naming its schema files, its
//! version and its dependencies. [`PackageRegistry`] loads packages from search paths, resolves
//! their dependencies, and merges their types into one environment with every type qualified by
//! its package name (`security-binary.Binary`), so two packages may define the same short name.
//!
//! Loading and resolution iterate `BTreeMap`s, so the order of packages, types and error
//! messages is the same run to run.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::{parse_schema, Schema, TypeDef};

// region:    --- Package Manifest

/// Package manifest (hel-package.toml)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageManifest {
    /// Package name (e.g., "security-binary")
    pub name: String,
    /// Semver version string
    pub version: String,
    /// List of schema files to load (in order) or glob pattern
    pub schemas: Vec<String>,
    /// Dependencies: package_name -> version_requirement
    #[serde(default)]
    pub dependencies: BTreeMap<String, String>,
    /// Optional built-ins namespace (defaults to package name)
    #[serde(default)]
    pub builtins_namespace: Option<String>,
}

impl PackageManifest {
    /// Parse manifest from TOML string
    ///
    /// # Errors
    ///
    /// Returns [`PackageError::ManifestParse`] if `content` is not valid TOML for a
    /// manifest, or if a required key (`name`, `version` or `schemas`) is absent.
    pub fn from_toml(content: &str) -> Result<Self, PackageError> {
        toml::from_str(content).map_err(|e| PackageError::ManifestParse(e.to_string()))
    }

    /// Load manifest from file
    ///
    /// # Errors
    ///
    /// Returns [`PackageError::Io`] if `path` cannot be read, or the same errors as
    /// [`PackageManifest::from_toml`] if its contents are not a valid manifest.
    pub fn from_file(path: &Path) -> Result<Self, PackageError> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            PackageError::Io(format!(
                "Failed to read manifest at {}: {}",
                path.display(),
                e
            ))
        })?;
        Self::from_toml(&content)
    }
}

// endregion: --- Package Manifest

// region:    --- Loaded Package

/// A loaded package with parsed schemas and metadata
#[derive(Debug, Clone)]
pub struct SchemaPackage {
    /// The manifest
    pub manifest: PackageManifest,
    /// Parsed schemas (combined)
    pub schema: Schema,
    /// Imports declared in schema files
    pub imports: Vec<String>,
    /// Package root directory
    pub root_path: PathBuf,
}

impl SchemaPackage {
    /// Load a package from a directory containing hel-package.toml
    ///
    /// Reads every schema file the manifest lists, in manifest order, and merges their
    /// types into one [`Schema`].
    ///
    /// # Errors
    ///
    /// Returns [`PackageError::Io`] if the manifest or any listed schema file cannot be
    /// read, [`PackageError::ManifestParse`] if the manifest is malformed,
    /// [`PackageError::SchemaParse`] if a schema file does not parse, and
    /// [`PackageError::DuplicateType`] if two schema files declare the same type name.
    pub fn from_directory(dir: &Path) -> Result<Self, PackageError> {
        let manifest_path = dir.join("hel-package.toml");
        let manifest = PackageManifest::from_file(&manifest_path)?;

        let mut combined_schema = Schema::new();
        let mut all_imports = Vec::new();

        for schema_file in &manifest.schemas {
            let schema_path = dir.join(schema_file);
            let content = std::fs::read_to_string(&schema_path).map_err(|e| {
                PackageError::Io(format!(
                    "Failed to read schema {}: {}",
                    schema_path.display(),
                    e
                ))
            })?;

            // Imports are collected from the raw text because `parse_schema` keeps only the
            // types; dependencies are resolved by the registry, not by the schema itself.
            let imports = extract_imports(&content);
            all_imports.extend(imports);

            let parsed = parse_schema(&content).map_err(|e| PackageError::SchemaParse {
                package: manifest.name.clone(),
                file: schema_file.clone(),
                error: e,
            })?;

            for (name, typedef) in parsed.types {
                if combined_schema.types.contains_key(&name) {
                    return Err(PackageError::DuplicateType {
                        package: manifest.name.clone(),
                        type_name: name.to_string(),
                    });
                }
                combined_schema.types.insert(name, typedef);
            }
        }

        Ok(Self {
            manifest,
            schema: combined_schema,
            imports: all_imports,
            root_path: dir.to_path_buf(),
        })
    }

    /// Get the namespace for this package (package name by default)
    #[must_use]
    pub fn namespace(&self) -> &str {
        &self.manifest.name
    }

    /// Get built-ins namespace (manifest.builtins_namespace or package name)
    #[must_use]
    pub fn builtins_namespace(&self) -> String {
        self.manifest
            .builtins_namespace
            .clone()
            .unwrap_or_else(|| self.manifest.name.clone())
    }
}

// endregion: --- Loaded Package

// region:    --- Package Registry

/// Registry that manages loading and resolving packages
#[derive(Debug, Clone)]
pub struct PackageRegistry {
    /// Search paths for packages
    search_paths: Vec<PathBuf>,
    /// Loaded packages: name -> package
    packages: BTreeMap<String, SchemaPackage>,
}

impl PackageRegistry {
    /// Create a new empty registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            search_paths: Vec::new(),
            packages: BTreeMap::new(),
        }
    }

    /// Add a search path for packages
    pub fn add_search_path(&mut self, path: PathBuf) {
        self.search_paths.push(path);
    }

    /// Load a package by name
    ///
    /// Searches in all registered search paths for a directory matching the package name.
    /// A manifest's declared version requirements are recorded but not enforced: `name`,
    /// not a version range, selects the package that is loaded.
    ///
    /// # Errors
    ///
    /// Returns [`PackageError::PackageNotFound`] if no search path holds a directory named
    /// `name` containing a `hel-package.toml`, [`PackageError::NameMismatch`] if the
    /// manifest in that directory declares a different name, and any error from
    /// [`SchemaPackage::from_directory`] while loading it.
    pub fn load_package(&mut self, name: &str) -> Result<&SchemaPackage, PackageError> {
        if self.packages.contains_key(name) {
            return Ok(&self.packages[name]);
        }

        let mut package_dir = None;
        for search_path in &self.search_paths {
            let candidate = search_path.join(name);
            if candidate.is_dir() && candidate.join("hel-package.toml").exists() {
                package_dir = Some(candidate);
                break;
            }
        }

        let dir = package_dir.ok_or_else(|| PackageError::PackageNotFound {
            name: name.to_string(),
            search_paths: self.search_paths.clone(),
        })?;

        let package = SchemaPackage::from_directory(&dir)?;

        if package.manifest.name != name {
            return Err(PackageError::NameMismatch {
                expected: name.to_string(),
                found: package.manifest.name.clone(),
            });
        }

        self.packages.insert(name.to_string(), package);
        Ok(&self.packages[name])
    }

    /// Resolve all dependencies for a root package recursively
    ///
    /// Returns packages in deterministic topological order (dependencies first).
    ///
    /// # Errors
    ///
    /// Returns [`PackageError::CircularDependency`] if the dependency graph contains a
    /// cycle, and otherwise any error from [`PackageRegistry::load_package`] for a package
    /// in the graph.
    pub fn resolve_all(&mut self, root_package: &str) -> Result<Vec<String>, PackageError> {
        let mut resolved = Vec::new();
        let mut visiting = std::collections::HashSet::new();

        self.resolve_recursive(root_package, &mut resolved, &mut visiting)?;

        Ok(resolved)
    }

    fn resolve_recursive(
        &mut self,
        package_name: &str,
        resolved: &mut Vec<String>,
        visiting: &mut std::collections::HashSet<String>,
    ) -> Result<(), PackageError> {
        // A package reached again while still being resolved is part of a cycle.
        if visiting.contains(package_name) {
            return Err(PackageError::CircularDependency {
                package: package_name.to_string(),
            });
        }

        if resolved.contains(&package_name.to_string()) {
            return Ok(());
        }

        visiting.insert(package_name.to_string());

        let package = self.load_package(package_name)?.clone();

        let deps: Vec<_> = package.manifest.dependencies.keys().cloned().collect();
        for dep in deps {
            self.resolve_recursive(&dep, resolved, visiting)?;
        }

        visiting.remove(package_name);
        resolved.push(package_name.to_string());

        Ok(())
    }

    /// Get a loaded package by name
    #[must_use]
    pub fn get_package(&self, name: &str) -> Option<&SchemaPackage> {
        self.packages.get(name)
    }

    /// Build a merged type environment from resolved packages
    ///
    /// Returns a map of qualified type names (`package.Type`) to their definitions.
    ///
    /// # Errors
    ///
    /// Returns [`PackageError::PackageNotFound`] if a name in `package_names` has not been
    /// loaded, and [`PackageError::TypeCollision`] if two packages contribute the same
    /// qualified type name.
    pub fn build_type_environment(
        &self,
        package_names: &[String],
    ) -> Result<TypeEnvironment, PackageError> {
        let mut types = BTreeMap::new();

        for pkg_name in package_names {
            let package =
                self.packages
                    .get(pkg_name)
                    .ok_or_else(|| PackageError::PackageNotFound {
                        name: pkg_name.clone(),
                        search_paths: self.search_paths.clone(),
                    })?;

            for (type_name, typedef) in &package.schema.types {
                let qualified_name = format!("{}.{}", package.namespace(), type_name);
                let qualified_name: Arc<str> = qualified_name.into();

                if types.contains_key(&qualified_name) {
                    return Err(PackageError::TypeCollision {
                        type_name: qualified_name.to_string(),
                    });
                }

                types.insert(qualified_name, typedef.clone());
            }
        }

        Ok(TypeEnvironment { types })
    }
}

impl Default for PackageRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// endregion: --- Package Registry

// region:    --- Type Environment

/// Merged type environment from multiple packages
#[derive(Debug, Clone)]
pub struct TypeEnvironment {
    /// Qualified type name (package.Type) -> TypeDef
    pub types: BTreeMap<Arc<str>, TypeDef>,
}

impl TypeEnvironment {
    /// Lookup a type by qualified name
    #[must_use]
    pub fn get_type(&self, qualified_name: &str) -> Option<&TypeDef> {
        self.types.get(qualified_name)
    }

    /// Validate all type references in the environment
    ///
    /// Unlike [`Schema::validate`](super::Schema::validate), which runs inside a single
    /// schema, this checks references against the merged, *qualified* namespace - a
    /// reference must be written `package.Type`, not a bare `Type`.
    ///
    /// # Errors
    ///
    /// Returns [`PackageError::UndefinedTypeReference`] for the first field whose type
    /// reference has no entry in the environment, naming the enclosing type as context.
    pub fn validate(&self) -> Result<(), PackageError> {
        for (qualified_name, typedef) in &self.types {
            for field in &typedef.fields {
                self.validate_field_type(&field.field_type, qualified_name)?;
            }
        }
        Ok(())
    }

    fn validate_field_type(
        &self,
        field_type: &super::FieldType,
        context: &str,
    ) -> Result<(), PackageError> {
        match field_type {
            super::FieldType::TypeRef(name) => {
                // Type references should be qualified (package.Type)
                if !self.types.contains_key(name) {
                    return Err(PackageError::UndefinedTypeReference {
                        type_name: name.to_string(),
                        context: context.to_string(),
                    });
                }
                Ok(())
            }
            super::FieldType::List(inner) | super::FieldType::Map(inner) => {
                self.validate_field_type(inner, context)
            }
            _ => Ok(()),
        }
    }
}

// endregion: --- Type Environment

// region:    --- Error Types

/// Package-related errors
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum PackageError {
    /// Manifest parsing error
    ManifestParse(String),
    /// Schema parsing error in a specific package/file
    SchemaParse {
        /// Name of the package whose manifest listed the file.
        package: String,
        /// Schema file that failed to parse, as named in the manifest.
        file: String,
        /// The underlying parse failure, with line information where available.
        error: String,
    },
    /// I/O error
    Io(String),
    /// Package not found in search paths
    PackageNotFound {
        /// Package name that was requested.
        name: String,
        /// Search paths that were consulted, for the error message.
        search_paths: Vec<PathBuf>,
    },
    /// Package name mismatch
    NameMismatch {
        /// Name that was requested.
        expected: String,
        /// Name the manifest actually declares.
        found: String,
    },
    /// Duplicate type in same package
    DuplicateType {
        /// Package declaring the type twice.
        package: String,
        /// The type name that appeared in more than one schema file.
        type_name: String,
    },
    /// Type collision across packages
    TypeCollision {
        /// The qualified name claimed by more than one package.
        type_name: String,
    },
    /// Undefined type reference
    UndefinedTypeReference {
        /// The unresolvable type name, as written.
        type_name: String,
        /// Qualified name of the type whose field holds the reference.
        context: String,
    },
    /// Circular dependency
    CircularDependency {
        /// Package that was reached again while its own dependencies were resolving.
        package: String,
    },
}

impl std::fmt::Display for PackageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PackageError::ManifestParse(e) => write!(f, "Failed to parse package manifest: {}", e),
            PackageError::SchemaParse {
                package,
                file,
                error,
            } => {
                write!(
                    f,
                    "Failed to parse schema in package '{}', file '{}': {}",
                    package, file, error
                )
            }
            PackageError::Io(e) => write!(f, "I/O error: {}", e),
            PackageError::PackageNotFound { name, search_paths } => {
                write!(
                    f,
                    "Package '{}' not found in search paths: {:?}",
                    name, search_paths
                )
            }
            PackageError::NameMismatch { expected, found } => {
                write!(
                    f,
                    "Package name mismatch: expected '{}', found '{}'",
                    expected, found
                )
            }
            PackageError::DuplicateType { package, type_name } => {
                write!(f, "Duplicate type '{}' in package '{}'", type_name, package)
            }
            PackageError::TypeCollision { type_name } => {
                write!(
                    f,
                    "Type name collision: '{}' is defined in multiple packages",
                    type_name
                )
            }
            PackageError::UndefinedTypeReference { type_name, context } => {
                write!(f, "Undefined type reference '{}' in {}", type_name, context)
            }
            PackageError::CircularDependency { package } => {
                write!(
                    f,
                    "Circular dependency detected involving package '{}'",
                    package
                )
            }
        }
    }
}

impl std::error::Error for PackageError {}

// endregion: --- Error Types

// region:    --- Import Parsing

/// Extract import declarations from schema content
///
/// Looks for lines like:
///   import "package-name";
///   import "security-binary";
fn extract_imports(content: &str) -> Vec<String> {
    let mut imports = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.starts_with("import ") {
            if let Some(rest) = line.strip_prefix("import ") {
                let rest = rest.trim().trim_end_matches(';').trim();
                if let Some(name) = rest.strip_prefix('"') {
                    if let Some(name) = name.strip_suffix('"') {
                        imports.push(name.to_string());
                    }
                }
            }
        }
    }

    imports
}

// endregion: --- Import Parsing

// region:    --- Tests

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_package(dir: &Path, name: &str, deps: &[(&str, &str)]) -> std::io::Result<()> {
        fs::create_dir_all(dir.join("schema"))?;

        let mut manifest = format!(
            r#"
name = "{}"
version = "0.1.0"
schemas = ["schema/00_domain.hel"]
"#,
            name
        );

        if !deps.is_empty() {
            manifest.push_str("\n[dependencies]\n");
            for (dep_name, dep_version) in deps {
                manifest.push_str(&format!("{} = \"{}\"\n", dep_name, dep_version));
            }
        }

        fs::write(dir.join("hel-package.toml"), manifest)?;

        let schema = format!(
            r#"
type {}Type {{
    value: String
}}
"#,
            name.replace('-', "_")
        );
        fs::write(dir.join("schema/00_domain.hel"), schema)?;

        Ok(())
    }

    #[test]
    fn test_package_manifest_parse() {
        let toml = r#"
name = "test-package"
version = "1.0.0"
schemas = ["schema/00_domain.hel"]

[dependencies]
other-package = "0.1.0"
"#;

        let manifest = PackageManifest::from_toml(toml).expect("parse failed");
        assert_eq!(manifest.name, "test-package");
        assert_eq!(manifest.version, "1.0.0");
        assert_eq!(manifest.schemas.len(), 1);
        assert_eq!(manifest.dependencies.len(), 1);
    }

    #[test]
    fn test_extract_imports() {
        let content = r#"
import "core-types";
import "security-binary";

type MyType {
    field: String
}
"#;

        let imports = extract_imports(content);
        assert_eq!(imports.len(), 2);
        assert_eq!(imports[0], "core-types");
        assert_eq!(imports[1], "security-binary");
    }

    #[test]
    fn test_package_loading() -> Result<(), Box<dyn std::error::Error>> {
        let temp = TempDir::new()?;
        let pkg_dir = temp.path().join("test-pkg");
        create_test_package(&pkg_dir, "test-pkg", &[])?;

        let package = SchemaPackage::from_directory(&pkg_dir)?;
        assert_eq!(package.manifest.name, "test-pkg");
        assert_eq!(package.schema.types.len(), 1);

        Ok(())
    }

    #[test]
    fn test_package_registry_loading() -> Result<(), Box<dyn std::error::Error>> {
        let temp = TempDir::new()?;
        let pkg_dir = temp.path().join("test-pkg");
        create_test_package(&pkg_dir, "test-pkg", &[])?;

        let mut registry = PackageRegistry::new();
        registry.add_search_path(temp.path().to_path_buf());

        let package = registry.load_package("test-pkg")?;
        assert_eq!(package.manifest.name, "test-pkg");

        Ok(())
    }

    #[test]
    fn test_dependency_resolution() -> Result<(), Box<dyn std::error::Error>> {
        let temp = TempDir::new()?;

        let base_dir = temp.path().join("base-pkg");
        create_test_package(&base_dir, "base-pkg", &[])?;

        let dep_dir = temp.path().join("dep-pkg");
        create_test_package(&dep_dir, "dep-pkg", &[("base-pkg", "0.1.0")])?;

        let mut registry = PackageRegistry::new();
        registry.add_search_path(temp.path().to_path_buf());

        let resolved = registry.resolve_all("dep-pkg")?;
        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved[0], "base-pkg"); // dependency first
        assert_eq!(resolved[1], "dep-pkg");

        Ok(())
    }

    #[test]
    fn test_type_environment_building() -> Result<(), Box<dyn std::error::Error>> {
        let temp = TempDir::new()?;
        let pkg_dir = temp.path().join("test-pkg");
        create_test_package(&pkg_dir, "test-pkg", &[])?;

        let mut registry = PackageRegistry::new();
        registry.add_search_path(temp.path().to_path_buf());

        let resolved = registry.resolve_all("test-pkg")?;
        let env = registry.build_type_environment(&resolved)?;

        // Type should be qualified as "test-pkg.test_pkgType"
        assert!(env.get_type("test-pkg.test_pkgType").is_some());

        Ok(())
    }

    #[test]
    fn test_circular_dependency_detection() -> Result<(), Box<dyn std::error::Error>> {
        let temp = TempDir::new()?;

        // Mutually dependent: pkg-a -> pkg-b -> pkg-a
        let a_dir = temp.path().join("pkg-a");
        create_test_package(&a_dir, "pkg-a", &[("pkg-b", "0.1.0")])?;

        let b_dir = temp.path().join("pkg-b");
        create_test_package(&b_dir, "pkg-b", &[("pkg-a", "0.1.0")])?;

        let mut registry = PackageRegistry::new();
        registry.add_search_path(temp.path().to_path_buf());

        let result = registry.resolve_all("pkg-a");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PackageError::CircularDependency { .. }
        ));

        Ok(())
    }
}

// endregion: --- Tests
