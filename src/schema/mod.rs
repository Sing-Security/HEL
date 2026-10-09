//! Schema definition support for HEL.
//!
//! Declarative definitions of a domain's types: a product declares its data model in `.hel`
//! schema files instead of implementing a resolver in Rust.

use std::collections::BTreeMap;
use std::sync::Arc;

pub mod package;
pub use package::{PackageError, PackageManifest, PackageRegistry, SchemaPackage, TypeEnvironment};

/// The declared type of a schema field
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum FieldType {
    /// A boolean.
    Bool,
    /// A string.
    String,
    /// A number, held as `f64` at evaluation time.
    Number,
    /// A homogeneous list whose elements all have the given type.
    List(Box<FieldType>),
    /// A homogeneous map whose values all have the given type.
    Map(Box<FieldType>),
    /// A reference to a type declared elsewhere in the same schema.
    ///
    /// Validated by [`Schema::validate`], which requires the named type to exist.
    TypeRef(Arc<str>),
}

/// A single field of a [`TypeDef`]
#[derive(Debug, Clone)]
pub struct FieldDef {
    /// Field name, as written in the schema.
    pub name: Arc<str>,
    /// Declared type, including the element type of `List`/`Map`.
    pub field_type: FieldType,
    /// Whether the field was declared with a `?` suffix and may be absent.
    pub optional: bool,
    /// Description of the field; [`parse_schema`] does not populate it.
    pub description: Option<Arc<str>>,
}

/// A named type in a [`Schema`]
#[derive(Debug, Clone)]
pub struct TypeDef {
    /// Type name; also the key this type is stored under in [`Schema::types`].
    pub name: Arc<str>,
    /// Fields in declaration order, which is preserved.
    pub fields: Vec<FieldDef>,
    /// Description of the type; [`parse_schema`] does not populate it.
    pub description: Option<Arc<str>>,
}

/// A set of named types
///
/// Types are held in a `BTreeMap` so iteration is sorted by name and therefore
/// deterministic across runs.
#[derive(Debug, Clone)]
pub struct Schema {
    /// The declared types, keyed by name.
    pub types: BTreeMap<Arc<str>, TypeDef>,
}

impl Schema {
    /// Create an empty schema
    #[must_use]
    pub fn new() -> Self {
        Self {
            types: BTreeMap::new(),
        }
    }

    /// Add a type definition to the schema
    pub fn add_type(&mut self, type_def: TypeDef) {
        self.types.insert(type_def.name.clone(), type_def);
    }

    /// Get a type definition by name
    #[must_use]
    pub fn get_type(&self, name: &str) -> Option<&TypeDef> {
        self.types.get(name)
    }

    /// Validate that all type references are defined
    ///
    /// Walks every field of every type, including the element types of `List`/`Map`, and
    /// checks each [`FieldType::TypeRef`] against the names in [`Schema::types`].
    ///
    /// # Errors
    ///
    /// Returns `Err` naming the first undefined type reference found. Iteration follows the
    /// `BTreeMap` order, so the name reported is stable for a given schema.
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
/// The syntax is simpler than the expression language and is read line by line:
///
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
///
/// Recognised primitives are `Bool`/`Boolean`, `String`, and `Number`/`Float`/`f64`;
/// `List<T>` and `Map<T>` nest; any other name is a [`FieldType::TypeRef`]. A `?` suffix on
/// a field name marks it [`optional`](FieldDef::optional). Blank lines, lines beginning with
/// `//` or `#`, and `import` lines (resolved by the package registry, not the schema) are
/// ignored, and trailing commas are tolerated. The finished schema is checked by
/// [`Schema::validate`], so an undefined type reference is a parse error.
///
/// # Errors
///
/// Returns `Err` if a `type` header is malformed, if a field line has neither `:` nor a
/// name, if a field's type reference has no matching declaration, if a type name is
/// declared twice, if a type block is never closed, or if a line outside a type block is
/// none of `type`, `}`, `import`, a comment or blank.
///
/// # Examples
///
/// ```
/// use hel::schema::parse_schema;
///
/// let schema = parse_schema("
/// type Contact {
///     email: String
/// }
///
/// type Lead {
///     contacts: List<Contact>
/// }
/// ").expect("schema should parse");
///
/// assert!(schema.get_type("Lead").is_some());
/// ```
pub fn parse_schema(input: &str) -> Result<Schema, String> {
    let mut schema = Schema::new();
    let mut current_type: Option<TypeDef> = None;

    for line in input.lines() {
        let line = line.trim();

        if line.is_empty() || line.starts_with("//") || line.starts_with('#') {
            continue;
        }

        // Package imports name other packages; the registry resolves them, not the schema.
        if line.starts_with("import ") {
            continue;
        }

        if line.starts_with("type ") {
            if let Some(type_def) = current_type {
                return Err(format!(
                    "Unclosed type block: '{}' was never closed with '}}'",
                    type_def.name
                ));
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
            continue;
        }

        if line == "}" {
            let type_def = current_type
                .take()
                .ok_or_else(|| "Unexpected '}' outside a type block".to_string())?;

            if schema.types.contains_key(&type_def.name) {
                return Err(format!("Duplicate type definition: {}", type_def.name));
            }
            schema.add_type(type_def);
            continue;
        }

        match current_type.as_mut() {
            Some(type_def) => {
                // A `?` suffix on the name is the only difference between a required and
                // an optional field.
                let field_line = line.trim_end_matches(',');
                let (field_name, rest) = if let Some(colon_pos) = field_line.find(':') {
                    (&field_line[..colon_pos], &field_line[colon_pos + 1..])
                } else {
                    return Err(format!("Invalid field definition: {}", line));
                };

                let (name, optional) =
                    if let Some(name_without_suffix) = field_name.strip_suffix('?') {
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
            None => return Err(format!("Unexpected content outside a type block: {}", line)),
        }
    }

    if let Some(type_def) = current_type {
        return Err(format!(
            "Unclosed type block: '{}' was never closed with '}}'",
            type_def.name
        ));
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

    #[test]
    fn test_parse_schema_rejects_duplicate_type() {
        let schema_text = r#"
type Lead {
    email: String
}

type Lead {
    phone: String
}
"#;

        let result = parse_schema(schema_text);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Duplicate type definition"));
    }

    #[test]
    fn test_parse_schema_rejects_unclosed_block_at_eof() {
        let result = parse_schema("type Lead {\n    email: String\n");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unclosed type block"));
    }

    #[test]
    fn test_parse_schema_rejects_unclosed_block_at_next_header() {
        let schema_text = r#"
type Lead {
    email: String

type Contact {
    name: String
}
"#;

        let result = parse_schema(schema_text);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unclosed type block"));
    }

    #[test]
    fn test_parse_schema_rejects_stray_content() {
        let result = parse_schema("email: String");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("outside a type block"));

        let result = parse_schema("}");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unexpected '}'"));
    }

    #[test]
    fn test_parse_schema_tolerates_import_lines() {
        let schema_text = r#"
import "core-types";

type Lead {
    email: String
}
"#;

        let schema = parse_schema(schema_text).expect("parse failed");
        assert_eq!(schema.types.len(), 1);
    }
}

// Additional integration tests
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_parse_binary_schema() {
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

        let schema = parse_schema(schema_text).expect("Failed to parse binary schema");
        assert!(schema.get_type("Binary").is_some());
        assert!(schema.get_type("Security").is_some());
        assert!(schema.get_type("Import").is_some());

        let import_type = schema.get_type("Import").unwrap();
        assert_eq!(import_type.fields.len(), 2);
        assert!(import_type.fields[1].optional); // library is optional
    }

    #[test]
    fn test_parse_crm_schema() {
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

        let schema = parse_schema(schema_text).expect("Failed to parse CRM schema");
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
