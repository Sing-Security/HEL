# Contributing to `hel` (Hermes Expression Language)

Status: OPEN (Apache-2.0)  
SPDX-License-Identifier: Apache-2.0

This document describes how to contribute to the `hel` crate. It is written for engineers working on the crate itself, authors of builtins and schema packages, and product integrators who will embed HEL in host systems. All contributions must follow the GEMINI platform rules: determinism, auditability, IP separation (open vs closed), minimal `unsafe`, and clear error handling.

Table of contents
- Purpose & scope
- Getting started
- Development guidelines
  - Determinism & safety
  - Error handling
  - `unsafe` usage
  - Public API & stability
- Code style & repository conventions
  - Rust10x & Cargo rules
  - Comment regions & section markers
  - Tests & examples
- Schema packages
- Builtins (open vs closed)
- Tracing & audit evidence
- CI, tests, and release process
- PR checklist (authors)
- Review checklist (maintainers)
- Contacts & escalation

---

Purpose & scope
- Keep platform concerns (language, parser, AST, evaluator primitives, schema loaders, builtin provider interfaces, trace formats) inside this crate.
- Do NOT place product-specific heuristics, scoring, or proprietary builtins inside this crate. Closed/product builtin implementations belong in separate closed crates and are injected via the `BuiltinsProvider` interface.
- Contributions may add: language fixes, parser improvements, schema loader fixes, open builtin helpers (only generic ones), trace improvements, documentation, tests, and tooling that improves determinism and auditability.

Getting started
1. Read the top-level language spec: `docs/hel.md`.
2. Read crate README: `README.md` (in this repository).
3. Read the developer docs in this directory: `INTRO.md`, `USAGE.md`, `SCHEMA.md`, `BUILTINS.md`, `TRACING.md`.
4. Run the test suite and linters locally (CI will run them too).
5. Open a small, focused PR that documents the rationale for changes and demonstrates reproducible tests.

Development guidelines

Determinism & safety
- Every evaluation path must be deterministic: identical inputs (expression text, schema, resolver inputs, builtin registry snapshot) must yield identical outputs and identical traces.
- No implicit I/O or reliance on environment variables during evaluation. If a builtin or evaluator requires time or randomness, the host must provide deterministic injection points (seeded RNG or pinned clock).
- Avoid use of APIs that can behave nondeterministically across platforms (e.g., iteration over `HashMap` without `BTreeMap` guarantees). Use deterministic containers for any data recorded in traces (`BTreeMap`, sorted vectors).
- Builtins that accept untrusted input must enforce bounded processing (input length limits, complexity limits, safe regex engines).

Error handling
- Public APIs MUST return `Result<T, E>` and never `panic!` in library code. Use the project error pattern described in the project docs (`error.rs` pattern).
- Parse errors must include span/location information; type/schema errors must be explicit and machine-readable where possible.
- Builtins must return structured `BuiltinError` with `code`, `message`, and optional `details` (map).

`unsafe` usage
- Minimize `unsafe`. If used:
  - Document *why* it is necessary and which invariants the `unsafe` block relies on.
  - Wrap `unsafe` in a safe, well-tested API boundary.
  - Add tests specifically covering the `unsafe`-backed functionality.

Public API & stability
- The crate is intended to be a stable platform for product integration. Avoid breaking changes to public types without bumping major versions and providing migration guides.
- Keep product-agnostic: do not export product rule packs or closed builtins.

Code style & repository conventions

Rust10x & Cargo rules
- Follow the Cargo.toml best practices in GEMINI (section in repo). Key notes:
  - Use `edition = "2024"` where appropriate.
  - Organize dependencies into sections with `# --` comments as described in the Cargo rules.
  - Keep `unsafe` forbids/lints documented in `lints.rust`.
- Use `rustfmt` and `clippy` defaults. Fix clippy lints unless the lint is intentionally suppressed with a comment explaining why.

Comment regions & section markers
- Use the prescribed region delimiters in source files:
  - `// region:    --- Modules` ... `// endregion: --- Modules` at top of `main.rs`, `lib.rs`, and `mod.rs`.
  - Use `// --` section markers within functions and large blocks.
- Keep regions short and purposeful. Tests must be wrapped in `// region:    --- Tests` and `// endregion: --- Tests` when inlined.

Tests & examples
- Tests must be deterministic. Use the test `Result<T>` alias pattern for test modules:
  - `type Result<T> = core::result::Result<T, Box<dyn std::error::Error>>;`
- Name test functions with the pattern: `test_<module_path>_<function_under_test>_<variant>()`.
- For example: `test_support_text_replace_markers_simple`.
- Group unit tests with `#[cfg(test)] mod tests { ... }` and follow the section markers.
- Integration tests and example programs go under `examples/` and must be small, focused, and runnable with `cargo run --example`.
- Test trace content in unit tests: assert presence of key `builtin_call` entries and expected `ValueSummary` outcomes.
- For floating point checks, use an explicit epsilon or `approx_eq` helper.

Schema packages
- Schema packages are versioned artifacts. Follow `SCHEMA.md` for package layout (`hel-package.toml`, `schema_files`) and deterministic merge order.
- Any change to a published schema file MUST be a new package version.
- Add CI checks to validate schema syntax and semantic correctness (type resolution).
- When authoring schema packages, include example fixture payloads and a README describing intended usage.

Builtins (open vs closed)
- Open builtins (generic operations) may live in the crate; they must be useful across products and fully documented, deterministic, and tested.
- Closed builtins must be implemented outside this crate and injected via `BuiltinsProvider`.
- When adding or modifying builtin provider interfaces:
  - Document the metadata fields (name, version, deterministic flag).
  - Ensure the `BuiltinsRegistry` has snapshot capabilities so hosts can persist registry state with traces.

Tracing & audit evidence
- The evaluator must emit a deterministic `EvalTrace` for each evaluation. See `TRACING.md`.
- Traces must include:
  - expression id/hash, expression text (or pointer), schema snapshot(s), builtin registry snapshot, ordered `entries`, and final `result`.
  - `builtin_call` steps must include builtin name, version, inputs (or input hashes for large blobs), output, and optional duration.
- Hosts must persist trace artifacts and snapshots for reproducibility. Add tests that assert trace format stability for small changes.

CI, tests, and release process
- CI must:
  - Run `cargo check`, `cargo test`, and `cargo clippy`.
  - Run schema validation, builtins tests, and trace JSON schema validation if present.
  - Run docs build (`cargo doc` or md lint) for the `docs/` directory and the crate README.
- Release procedure:
  - Bump crate version according to semver.
  - Tag the commit and record changelog entries describing deterministic behavior changes, builtin metadata changes, or trace format changes.
  - If trace or public API changes, update `trace_format_version` and document migration steps.

PR checklist (authors)
Before opening a PR, ensure:
- [ ] The change is small and focused.
- [ ] All new behavior is covered by unit tests (parser, evaluator, builtin unit, trace assertions).
- [ ] `cargo fmt` and `cargo clippy` pass (or intentional lints are documented).
- [ ] Documentation updated: README, docs/*.md, and reference pieces as appropriate.
- [ ] Schema changes (if any) use a new package version and include fixtures.
- [ ] Builtins changes include metadata and tests demonstrating deterministic results.
- [ ] Trace changes include a JSON Schema update and test(s) asserting canonical serialization where applicable.
- [ ] No product-specific logic or closed builtins are added to this open crate.

Review checklist (maintainers)
- [ ] Verify determinism: can the change cause nondeterministic evaluation? (iter order, RNG, time)
- [ ] Verify auditability: are spans, traces, and metadata produced and stable?
- [ ] Verify API stability: is any public API changed? If yes, is the change documented and versioned?
- [ ] Security: does the change introduce unsafe code or new dependency risks? Ensure `unsafe` is justified and reviewed.
- [ ] Documentation: does PR update developer docs and reference material?
- [ ] Tests: do tests exercise edge cases and trace outcomes? Are CI artifacts green?
- [ ] Dependency review: new dependencies must be reviewed for license compatibility and SBOM inclusion.

Contacts & escalation
- For design or policy questions (determinism, trace format, open/closed boundary), raise an issue and tag `@platform-arch` and `@security-leads`.
- For urgent security or reproducibility issues, open a high-priority issue and notify the code owners directly.

Appendix — Useful patterns & references
- Error pattern: include `Error` enum + `pub type Result<T> = core::result::Result<T, Error>` in crate root; follow the `error.rs` pattern in the GEMINI docs.
- Comment regions: use `// region:    --- Name` and `// endregion: --- Name`.
- Tests naming & structure: use the Rust10x test structure with `// -- Setup & Fixtures`, `// -- Exec`, `// -- Check`.
- Cargo.toml organization: follow Cargo best practice sections and include `# -- Others` when `derive_more` or similar is present.

Thank you for contributing. Keep changes focused, deterministic, and auditable — the platform depends on reproducible behavior and clear evidence to support compliance and long-term product integrity.
