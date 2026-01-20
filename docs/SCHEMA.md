# HEL — Schema Packages, Format, and Loaders

This document describes HEL's schema package format, how to author schema files, and the recommended patterns for loading, validating, and merging schema packages in host applications. It is targeted at engineers who will author domain schemas, implement schema-backed resolvers, or integrate schema package loading into their runtime.

Goals
- Define the package manifest and directory layout for HEL schema packages.
- Describe the HEL schema language (types, optional fields, collections).
- Explain loader/merge semantics and validation rules.
- Provide best practices for versioning, auditability, and deterministic resolution.

Audience
- You are embedding HEL and need to define domain types for host facts.
- You are authoring or reviewing schema packages used by product rule sets.
- You are implementing schema loaders and resolvers that map host data to HEL `Value`s.

Principles
- Determinism: Schema loading and merging is deterministic and reproducible.
- Auditability: Every schema package is versioned and its manifest recorded with evaluation evidence.
- Separation: Schema packages describe surface types only; no product logic or heuristics belong in schema files.
- Validation: Loaders must validate references and types; failure should yield structured errors with span/manifest context.

Table of contents
- Package layout and manifest
- Schema language (types, optional fields, generics)
- Example schema package
- Loading and merging semantics
- Validation rules
- Resolver mapping guidance
- Versioning and packaging
- Testing and CI recommendations
- Audit & evidence capture

---

## Package layout and manifest

A schema package is a directory with a manifest (`hel-package.toml`) and one or more `.hel` type definition files.

Recommended layout:
- `hel-package.toml` — required package manifest
- `schema/` — directory containing one or more `.hel` schema files
  - `00_types.hel`
  - `10_enrichment.hel`
- `README.md` — optional human-readable explanation and scope
- `LICENSE` — package license (recommended)

Example `hel-package.toml` (semantic fields you must include):
- `name` — package name (string, reverse-domain recommended)
- `version` — semver string
- `description` — short description
- `authors` — optional list
- `schema_files` — ordered list of schema file paths (deterministic order for merging)

A conceptual manifest example:
```/dev/null/hel-package.toml#L1-20
[package]
name = "com.example.product.schema"
version = "0.1.0"
description = "Domain schema for Example Product facts"
authors = ["Example Team <security@example.com>"]

# Deterministic merge order for schema files.
schema_files = ["schema/00_types.hel", "schema/10_enrichment.hel"]
```

Loaders MUST use `schema_files` order to merge and resolve type references deterministically.

---

## Schema language (overview)

The schema language is intentionally small and structural. It describes named types, fields, optional fields, lists, and maps. The language is designed to be easy to author and parse.

Core constructs
- `type Name { field: Type, ... }` — record/object types
- Primitive names: `Bool`, `String`, `Number` (float-64), `Bytes`, `Time`
- Collections: `List<T>`, `Map<String, T>`
- Optional fields: `field?: Type` (or `field: Type?`) — field may be absent
- Type references: reference other `type` names defined in the package or imported packages

Example type file:
```/dev/null/schema/00_types.hel#L1-60
# Basic host facts for a binary analysis product
type BinaryInfo {
    format: String
    arch: String
    entry_point: Number
    size: Number
    is_stripped?: Bool   # optional flag
    sections: List<SectionInfo>
}

type SectionInfo {
    name: String
    size: Number
    is_executable: Bool
}
```

Notes
- Use `List<T>` for homogeneous ordered sequences.
- Use `Map<String, T>` for dictionary-like structures. Keys are strings.
- Optional fields are represented in the runtime `Value` model as `null` when missing; resolvers should return `None` for missing attributes.

---

## Example schema package

A minimal package with binary facts and enrichment:

```/dev/null/schema_package_example/tree#L1-120
# hel-package.toml
[package]
name = "com.example.binary.schema"
version = "0.1.0"
schema_files = ["schema/00_binary.hel", "schema/10_enrichment.hel"]

# schema/00_binary.hel
type BinaryInfo {
    format: String
    arch: String
    entry_point: Number
    size: Number
    sections: List<SectionInfo>
}

type SectionInfo {
    name: String
    size: Number
    is_executable: Bool
}

# schema/10_enrichment.hel
type Enrichment {
    compiler: String?
    build_id: String?
    detected: Map<String, String>  # e.g., {"packer": "upx"}
}
```

Authors should document example JSON/Value payloads in the README for consumers of the schema.

---

## Loading and merging semantics

The crate exposes loaders to read `hel-package.toml` and the listed `schema_files`. Loaders must:
1. Read `hel-package.toml`.
2. Validate `schema_files` presence and deterministic order.
3. Parse `.hel` files in the order specified.
4. Merge type declarations: later files can reference earlier types; duplicates are an error (no silent overrides).
5. Produce a `Schema` object that includes:
   - Type definitions
   - Source file -> line range mapping for diagnostics (for auditing)
   - Canonical package identifier (`name@version`)
6. Optionally allow explicit imports from other packages (future extension). For now, keep package scopes independent unless host merges multiple packages.

Merge rules (deterministic)
- Duplicate type names in the same package → validation error.
- Field name conflicts across merges (if we ever support cross-package merges) must be resolved via explicit package-qualified names; avoid implicit name collisions.

---

## Validation rules

Loaders must validate:
- Manifest presence and semantic correctness (semver parse).
- All `schema_files` exist and are readable.
- Syntax correctness for each `.hel` file.
- Type references resolve to declared types within the merged package.
- No duplicate type declarations.
- Field types are valid (primitives or declared types, or parameterized collections).
- Warn on unused types (optional check).
- Provide structured diagnostics: file path, line/column, and a machine-readable error code.

Errors should be returned as typed `Result<Schema, SchemaError>` and include spans where applicable.

---

## Resolver mapping guidance

The `Schema` primarily guides host resolvers and tooling. When mapping host data to HEL `Value`s:
- Use `BTreeMap` for maps to guarantee deterministic ordering.
- For optional fields:
  - Missing field → resolver returns `None` → HEL receives `Null`.
  - Present field with value → convert to HEL `Value` according to schema.
- For lists: preserve deterministic order; if host source order is nondeterministic, define canonical ordering during ingestion.
- Type mismatches: resolvers may coerce where safe (e.g., integer -> Number float) but should surface warnings; prefer explicit, deterministic conversions.

Resolver contracts
- Must be free of side effects during `resolve_attr` calls.
- Should be performant and avoid unbounded recursion.
- Should expose an adapter for bulk mapping: convert host object to a `Map<String, Value>` once per evaluation rather than performing many resolver calls if host cost is high.

Example mapping note (conceptual):
```/dev/null/example_resolver_notes#L1-40
# For a `BinaryInfo` host struct:
- map `format` -> Value::String
- map `arch` -> Value::String
- map `entry_point` -> Value::Number (float)
- map `sections` -> Value::List of Map values (each SectionInfo)
```

---

## Versioning and packaging

- Each package MUST have a semver `version` in `hel-package.toml`.
- Host systems must record the exact `name@version` used for each evaluation for auditability.
- When changing a package:
  - Add a new semver version.
  - Do not mutate published package versions.
- Canonical packaging may be a tarball or directory; loaders should accept both unpacked directories and package archives.

---

## Testing and CI recommendations

- Unit test schema parsing with both valid and invalid examples to exercise diagnostics.
- Include sample host payloads (JSON) and round-trip tests: serialize -> resolve -> evaluate example HEL expressions.
- For each package version, publish automated validation that the package loads and that all declared types resolve in sample fixtures.
- Include trace-based tests to ensure resolver-to-HEL conversions include expected intermediate values.

---

## Audit & evidence capture

For every evaluation that depends on schema packages, record:
- The package manifest (`hel-package.toml`) content or its canonical hash (`sha256`).
- The schema file contents and their file-level checksums.
- The `Schema` version string and the loader/version that created the `Schema`.
- Any schema validation warnings or errors observed during load.

This information is necessary to reproduce evaluation results and to satisfy compliance requirements.

---

## Best practices summary

- Keep schema files small and focused; use multiple files with deterministic `schema_files` order.
- Version every package; never mutate published versions.
- Prefer `List<T>` and `Map<String,T>` for collections; prefer `BTreeMap` ordering in resolvers.
- Ensure resolvers are deterministic and avoid I/O during evaluation.
- Record manifest and schema checksums with evaluation traces for auditability.
- Validate packages in CI and include example fixtures per package version.
