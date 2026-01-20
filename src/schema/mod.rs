//! Schema definition support for HEL
//!
//! This module provides declarative schema definitions for domain types,
//! allowing products to define their data models in .hel schema files
//! instead of implementing resolvers in Rust code.

use std::collections::BTreeMap;
use std::sync::Arc;

pub mod package;
pub use package::{PackageError, PackageManifest, PackageRegistry, SchemaPackage, TypeEnvironment};

/// Field type definition
#[derive(Debug, Clone, PartialEq)]
pub enum FieldType {
	Bool,
	String,
	Number,
	List(Box<FieldType>),
	Map(Box<FieldType>),
	/// Reference to another type
	TypeRef(Arc<str>),
}

/// Field definition in a schema
#[derive(Debug, Clone)]
pub struct FieldDef {
	pub name: Arc<str>,
	pub field_type: FieldType,
	pub optional: bool,
	pub description: Option<Arc<str>>,
}

/// Type definition in a schema
#[derive(Debug, Clone)]
pub struct TypeDef {
	pub name: Arc<str>,
	pub fields: Vec<FieldDef>,
	pub description: Option<Arc<str>>,
}

/// Schema definition containing all types
#[derive(Debug, Clone)]
pub struct Schema {
	pub types: BTreeMap<Arc<str>, TypeDef>,
}

impl Schema {
	/// Create an empty schema
	pub fn new() -> Self {
		Self { types: BTreeMap::new() }
	}

	/// Add a type definition to the schema
	pub fn add_type(&mut self, type_def: TypeDef) {
		self.types.insert(type_def.name.clone(), type_def);
	}

	/// Get a type definition by name
	pub fn get_type(&self, name: &str) -> Option<&TypeDef> {
		self.types.get(name)
	}

	/// Validate that all type references are defined
	pub fn validate(&self) -> Result<(), String> {
		for type_def in self.types.values() {
			for field in &type_def.fields {
				self.validate_field_type(&field.field_type)?;
			}
		}
		Ok(())
	}

	fn validate_field_type(&self, field_type: &FieldType) -> Result<(), String> {
		match field_type {
			FieldType::TypeRef(name) => {
				if !self.types.contains_key(name) {
					return Err(format!("Undefined type reference: {}", name));
				}
				Ok(())
			}
			FieldType::List(inner) | FieldType::Map(inner) => self.validate_field_type(inner),
			_ => Ok(()),
		}
	}
}

impl Default for Schema {
	fn default() -> Self {
		Self::new()
	}
}

/// Parse a schema from HEL schema syntax
///
/// Schema files use a simplified syntax:
/// ```hel
/// type Lead {
///     vertical: String
///     stage: String
///     score: Number
///     contacts: List<Contact>
/// }
///
/// type Contact {
///     email: String
///     name: String
/// }
///
/// type Enrichment {
///     confidence: Number
///     source: String
///     data: Map<String>
/// }
/// ```
pub fn parse_schema(input: &str) -> Result<Schema, String> {
	let mut schema = Schema::new();
	let mut current_type: Option<TypeDef> = None;
	let mut in_type_block = false;

	for line in input.lines() {
		let line = line.trim();

		// Skip empty lines and comments
		if line.is_empty() || line.starts_with("//") || line.starts_with('#') {
			continue;
		}

		// Type definition start
		if line.starts_with("type ") {
			// Save previous type if any
			if let Some(type_def) = current_type.take() {
				schema.add_type(type_def);
			}

			let parts: Vec<&str> = line.split_whitespace().collect();
			if parts.len() < 3 || parts[2] != "{" {
				return Err(format!("Invalid type definition: {}", line));
			}

			current_type = Some(TypeDef {
				name: parts[1].into(),
				fields: Vec::new(),
				description: None,
			});
			in_type_block = true;
			continue;
		}

		// Type block end
		if line == "}" {
			if let Some(type_def) = current_type.take() {
				schema.add_type(type_def);
			}
			in_type_block = false;
			continue;
		}

		// Field definition
		if in_type_block && current_type.is_some() {
			if let Some(type_def) = current_type.as_mut() {
				// Parse field: name: Type or name?: Type for optional
				let field_line = line.trim_end_matches(',');
				let (field_name, rest) = if let Some(colon_pos) = field_line.find(':') {
					(&field_line[..colon_pos], &field_line[colon_pos + 1..])
				} else {
					return Err(format!("Invalid field definition: {}", line));
				};

				let (name, optional) = if let Some(name_without_suffix) = field_name.strip_suffix('?') {
					(name_without_suffix, true)
				} else {
					(field_name, false)
				};

				let type_str = rest.trim();
				let field_type = parse_field_type(type_str)?;

				type_def.fields.push(FieldDef {
					name: name.trim().into(),
					field_type,
					optional,
					description: None,
				});
			}
		}
	}

	// Save last type if any
	if let Some(type_def) = current_type {
		schema.add_type(type_def);
	}

	schema.validate()?;
	Ok(schema)
}

fn parse_field_type(type_str: &str) -> Result<FieldType, String> {
	let type_str = type_str.trim();

	// List<T>
	if type_str.starts_with("List<") && type_str.ends_with('>') {
		let inner = &type_str[5..type_str.len() - 1];
		let inner_type = parse_field_type(inner)?;
		return Ok(FieldType::List(Box::new(inner_type)));
	}

	// Map<T>
	if type_str.starts_with("Map<") && type_str.ends_with('>') {
		let inner = &type_str[4..type_str.len() - 1];
		let inner_type = parse_field_type(inner)?;
		return Ok(FieldType::Map(Box::new(inner_type)));
	}

	// Primitive types
	match type_str {
		"Bool" | "Boolean" => Ok(FieldType::Bool),
		"String" => Ok(FieldType::String),
		"Number" | "Float" | "f64" => Ok(FieldType::Number),
		// Type reference
		_ => Ok(FieldType::TypeRef(type_str.into())),
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_parse_simple_schema() {
		let schema_text = r#"
type Lead {
    vertical: String
    score: Number
}
		"#;

		let schema = parse_schema(schema_text).expect("parse failed");
		assert_eq!(schema.types.len(), 1);

		let lead_type = schema.get_type("Lead").expect("Lead type not found");
		assert_eq!(lead_type.fields.len(), 2);
		assert_eq!(lead_type.fields[0].name.as_ref(), "vertical");
		assert_eq!(lead_type.fields[1].name.as_ref(), "score");
	}

	#[test]
	fn test_parse_schema_with_lists() {
		let schema_text = r#"
type Contact {
    email: String
}

type Lead {
    contacts: List<Contact>
}
		"#;

		let schema = parse_schema(schema_text).expect("parse failed");
		assert_eq!(schema.types.len(), 2);

		let lead_type = schema.get_type("Lead").expect("Lead type not found");
		assert_eq!(lead_type.fields.len(), 1);

		match &lead_type.fields[0].field_type {
			FieldType::List(inner) => match inner.as_ref() {
				FieldType::TypeRef(name) => assert_eq!(name.as_ref(), "Contact"),
				_ => panic!("Expected TypeRef"),
			},
			_ => panic!("Expected List type"),
		}
	}

	#[test]
	fn test_parse_schema_with_optional() {
		let schema_text = r#"
type Lead {
    email: String
    phone?: String
}
		"#;

		let schema = parse_schema(schema_text).expect("parse failed");
		let lead_type = schema.get_type("Lead").expect("Lead type not found");

		assert!(!lead_type.fields[0].optional);
		assert!(lead_type.fields[1].optional);
	}

	#[test]
	fn test_schema_validation() {
		let schema_text = r#"
type Lead {
    contact: UnknownType
}
		"#;

		let result = parse_schema(schema_text);
		assert!(result.is_err());
		assert!(result.unwrap_err().contains("Undefined type reference"));
	}
}

// Additional integration tests
#[cfg(test)]
mod integration_tests {
	use super::*;

	#[test]
	fn test_parse_desmond_schema() {
		// This would be loaded from products/Desmond/schema/00_domain.hel in production
		let schema_text = r#"
type Binary {
    format: String
    arch: String
    entry_point: Number
    file_size: Number
}

type Security {
    pie: String
    nx: String
}

type Import {
    symbol: String
    library?: String
}
"#;

		let schema = parse_schema(schema_text).expect("Failed to parse Desmond schema");
		assert!(schema.get_type("Binary").is_some());
		assert!(schema.get_type("Security").is_some());
		assert!(schema.get_type("Import").is_some());

		let import_type = schema.get_type("Import").unwrap();
		assert_eq!(import_type.fields.len(), 2);
		assert!(import_type.fields[1].optional); // library is optional
	}

	#[test]
	fn test_parse_fidelis_schema() {
		// Example CRM/lead schema
		let schema_text = r#"
type Lead {
    vertical: String
    score: Number
    contacts: List<Contact>
}

type Contact {
    email: String
    name: String
    title?: String
}

type Enrichment {
    confidence: Number
    data: Map<String>
}
"#;

		let schema = parse_schema(schema_text).expect("Failed to parse Fidelis schema");
		assert!(schema.get_type("Lead").is_some());
		assert!(schema.get_type("Contact").is_some());
		assert!(schema.get_type("Enrichment").is_some());

		let lead_type = schema.get_type("Lead").unwrap();
		// Verify contacts field is List<Contact>
		match &lead_type.fields[2].field_type {
			FieldType::List(inner) => match inner.as_ref() {
				FieldType::TypeRef(name) => assert_eq!(name.as_ref(), "Contact"),
				_ => panic!("Expected TypeRef"),
			},
			_ => panic!("Expected List type"),
		}
	}
}
