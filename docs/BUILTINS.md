# HEL — Builtins (Provider & Registry)

This document explains how HEL's builtin function extension points work, how to implement open vs closed builtins, how to register them with the engine, and the rules you must follow to keep builtins deterministic, auditable, and safe. It is written for engineers who will:

- implement host/product-specific (closed) builtins,
- maintain or extend open builtins in the public crate,
- write tests for builtins and integration tests that assert trace contents.

Design goals (recap)
- Determinism: identical inputs → identical outputs.
- Safety: no I/O, no hidden globals during evaluation.
- Bounded execution: no unbounded recursion or super-linear regex backtracks.
- Auditability: every builtin call must be reflected in the evaluation trace with inputs, outputs, version, and a reference to implementation identity.
- Open/Closed boundary: the open crate should contain generic, broadly useful builtins only. Product-specific or proprietary functionality must be implemented in closed crates and injected at runtime.

Summary
- Builtins are functions available to HEL expressions (e.g., `entropy(bytes)`, `hash.sha256(bytes)`, `regex.is_match(re, s)`).
- The crate exposes two extension primitives:
  - `BuiltinsRegistry` — runtime registry that holds available builtin functions and their metadata.
  - `BuiltinsProvider` — an implementation a host can register with the `BuiltinsRegistry` to supply additional (closed) builtins.
- Builtin function implementations must be pure, deterministic, bounded, and return structured errors instead of panics.

Conceptual API notes
- The following sketches are conceptual pseudocode to communicate intent and are not intended to be taken as exact signatures — prefer the public types exported by the crate (`BuiltinsProvider`, `BuiltinsRegistry`, `BuiltinFn`, `CoreBuiltinsProvider`).

```/dev/null/builtin_signature.example#L1-80
// Conceptual shape of a builtin function
// fn builtin_fn(args: &[Value], ctx: &BuiltinContext) -> Result<Value, BuiltinError>
//
// Where:
// - `Value` is the HEL runtime value enum (Null, Bool, Number, String, List, Map, ...).
// - `BuiltinContext` is a read-only context supplied by the evaluator (for tracing, metadata).
// - `BuiltinError` is a structured error type with codes and message.
//
// The BuiltinsProvider interface lets the host enumerate builtin entries:
// trait BuiltinsProvider {
//     fn list_builtins(&self) -> Vec<BuiltinMetadata>;
//     fn get_builtin(&self, name: &str) -> Option<BuiltinFn>;
// }
```

Naming, metadata, and versioning
- Each builtin must be registered with metadata that includes:
  - canonical name (dot-separated, e.g., `hash.sha256`, `regex.is_match`),
  - short description,
  - version string (semver or revision hash),
  - deterministic guarantee flag (bool),
  - declared argument types/arity (optional, for fast validation).
- The registry should expose a way to serialize the set of registered builtin names + versions; the host must persist this with evaluation evidence so auditors can determine exactly which builtin implementations contributed to a result.

Open vs Closed builtins
- Open builtins: implemented in the `hel` crate (or other public ForgeCore crates). They must be:
  - Generic and broadly useful (strings, lists, deterministic hashing, collection helpers).
  - Fully documented and tested.
  - Free of product-specific heuristics.
- Closed builtins: implemented by host/product crates. They must be:
  - Packaged separately (different crate / binary module).
  - Registered at runtime via a `BuiltinsProvider` implementation.
  - Documented externally by the product (interface only in the open crate).
- The `BuiltinsRegistry` must accept attaching one or more `BuiltinsProvider` instances; providers may override names only if the registry is configured to allow explicit provider shadowing — prefer explicit rejection of collisions.

Trace integration (auditability)
- Every builtin call executed during evaluation must produce a trace entry that is recorded in the `EvalTrace`. A trace entry for a builtin call must include:
  - builtin name,
  - builtin version (from metadata),
  - input values (serialized deterministically),
  - output value or error (serialized deterministically),
  - optional execution cost estimate or measured duration (host may include),
  - expression span (where the call originated).
- The evaluator is responsible for producing the trace entry; builtin implementations should be given a `BuiltinContext` object that allows them to add structured diagnostics (not perform I/O).
- The trace format must be stable and serializable in audit artifacts.

Determinism & bounded execution
- Builtins must not perform external I/O during evaluation.
- Long-running or resource-heavy operations must be avoided or explicitly bounded (e.g., limit regex input size, limit list iteration counts, refuse operations which exceed configured limits).
- Regex builtins must use or mimic RE2-style semantics (no backtracking that can result in catastrophic runtime). If a closed builtin needs to use a regex engine with backtracking, the provider must enforce input length and complexity limits and document them.
- Builtins that require randomness must accept a deterministic seed via the `BuiltinContext` (provided by the host) or be disallowed in production.
- Avoid global mutable state. If internal caching is necessary, it must be guarded for determinism (e.g., keyed caches that only depend on input).

Error handling
- Builtins return structured errors (e.g., `BuiltinError { code: String, message: String, details: Option<Map> }`).
- Do not `panic!` in builtin implementations. Any unexpected condition should be converted to a `BuiltinError` and returned.
- The evaluator should convert builtin errors into evaluation errors with spans and include partial trace data.

Registration patterns & examples
- Typical registration steps a host will perform at startup:
  1. Create a `BuiltinsRegistry` instance used for all evaluations.
  2. Register built-in open functions exported by the `hel` crate (core builtins).
  3. Attach one or more closed `BuiltinsProvider` implementations provided by the product.
  4. When evaluating, pass the registry to the evaluator so builtin resolution happens via the registry.

Conceptual registration example (pseudocode):

```/dev/null/registry_example.rs#L1-120
// PSEUDOCODE (conceptual)
let mut registry = BuiltinsRegistry::new();

// register open/core builtins provided by the crate
registry.register_provider(Box::new(CoreBuiltinsProvider::default()));

// attach product-specific builtins implemented in a closed crate
let provider = Box::new(MyProductBuiltins::new(my_config));
registry.register_provider(provider);

// optionally list available builtins and their versions for evidence:
let registry_snapshot = registry.snapshot_meta(); // returns list of (name, version)
```

Implementing a closed builtin (recommended pattern)
- Implement a small type that implements `BuiltinsProvider` and returns metadata + function hooks.
- Keep the implementation small and focused; rely on well-audited libraries for heavy lifting when necessary (but still enforce deterministic constraints).
- Provide a public changelog and version tag for the provider so hosts can record provider versions in audit artifacts.

Pseudocode: minimal provider

```/dev/null/minimal_provider.rs#L1-160
// PSEUDOCODE (conceptual)
struct MyProductBuiltins {
    version: String,
    // optional configuration, caches, etc.
}

impl BuiltinsProvider for MyProductBuiltins {
    fn list_builtins(&self) -> Vec<BuiltinMetadata> {
        vec![
            BuiltinMetadata::new("product.special_score", "0.1.0", "Deterministic score"),
        ]
    }

    fn get_builtin(&self, name: &str) -> Option<BuiltinFn> {
        match name {
            "product.special_score" => Some(Box::new(|args, ctx| {
                // Validate args
                // Compute deterministic score
                // Return Result<Value, BuiltinError>
            })),
            _ => None,
        }
    }
}
```

Testing builtins
- Unit tests:
  - Test each builtin function on representative inputs, including edge cases (empty inputs, nulls, extremely large inputs).
  - Test deterministic behavior: same input yields same output.
  - Test error conditions and structured error content.
- Integration tests:
  - Evaluate HEL expressions which call the builtin and assert evaluation result and the presence and structure of trace entries (builtin name, inputs, outputs).
  - For closed builtins using non-deterministic resources (e.g., system time), test with deterministic injection (mocked `BuiltinContext`).
- Fuzzing:
  - Where applicable, fuzz input values but enforce maximum input sizes and complexity limits in the provider.
- CI:
  - Include fuzz/size checks that ensure new builtins do not accept unbounded input sizes without explicit limits.

Performance & resource control
- Builtins that process large binary blobs (e.g., `entropy`, `hash`) should be implemented to operate in streaming or bounded-memory modes where feasible.
- Provide configurable limits at provider creation time (e.g., maximum bytes to hash, maximum list iterations).
- The registry or the evaluator may expose a global resource budget for evaluation; builtins should respect the budget provided in `BuiltinContext`.

Packaging, upgrade, and audit practices
- Providers should include an explicit version string and ideally a changelog.
- Hosts must persist (with each evaluation record) the registry snapshot — builtin names with versions — so historical evaluations are reproducible.
- When upgrading closed builtins, treat as a potential behavioral change: record both old and new provider versions and re-run critical rule suites as part of the upgrade verification.

Suggested builtin metadata fields
- name: canonical name (string)
- version: semver or VCS hash (string)
- description: short text (string)
- deterministic: bool
- stable: bool (indicates API/semantics stability)
- params: optional signature description (arity, types)
- cost_estimate: optional numeric cost estimate for scheduling/budgeting

Security considerations
- Builtins must sanitize inputs (untrusted expression inputs may contain hostile data).
- Avoid unsafe native FFI unless absolutely necessary and encapsulate it inside a safe shim with limits.
- If using regexes, enforce input length/complexity caps and prefer RE2-like engines.
- Document any cryptographic operations and required libraries (and link to SBOM entries for the provider).

Operational checklist for hosts
- On startup:
  - Build `BuiltinsRegistry`.
  - Register `CoreBuiltinsProvider`.
  - Attach closed `BuiltinsProvider` implementations.
  - Snapshot registry metadata and store it in a retrievable location (e.g., service metadata endpoint).
- Prior to evaluation:
  - Ensure resolver is deterministic and free of side-effects.
  - Provide `BuiltinContext` that contains trace collector, resource/budget controls, and deterministic seeds (if required).
- After evaluation:
  - Persist evaluation trace, source text (or canonical hash), schema package versions, and registry snapshot.

Common builtin candidates (open crate)
- Strings: `str.len`, `str.contains`, `str.lower`, `str.upper`, `str.trim`
- Collections: `list.len`, `list.map`, `list.filter`, `list.any`, `list.all`, `list.unique`
- Hashing: `hash.sha256(bytes)`, `hash.sha1(bytes)`, `hash.md5(bytes)` (documented caveats)
- Byte analysis: `entropy(bytes) -> Number`, `bytes.slice(start, len)`
- Regex helpers: `regex.compile(pattern, flags) -> result<regex, err>`, `regex.is_match(regex, s)`
- IP/CIDR: `ip.parse`, `cidr.parse`, `ip.in_cidr`
- Time: `time.parse`, `time.diff` — only when host provides deterministic clock injection

Appendix: Example trace entry (recommended shape)
```json
{
  "type": "builtin_call",
  "builtin": "hash.sha256",
  "builtin_version": "0.2.1",
  "span": { "start": 12, "end": 28 },
  "inputs": [ "0xdeadbeef..." ],
  "output": "0x9f86d081884c7d659a2feaa0c55ad015",
  "status": "ok",
  "duration_ms": 0.3
}
