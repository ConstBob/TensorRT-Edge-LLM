# Repository Review Rules

## Review Scope

- Review against files in this repository and the context files selected by `.greptile/files.json`.
- Treat `.greptile/` as cascading configuration: root rules are shared defaults and folder rules add owner-specific expectations.
- For non-trivial changes, require matching documentation, validation commands, and a clear release-impact statement.

## Source Hygiene

- Do not add concrete review numbers, bug tracker numbers, internal-only model or checkpoint names, unreleased identifiers, internal project names, or deployment details to source code, public docs, tests, examples, or config.
- Tracking identifiers belong in review metadata, not repository files.
- If private-only content is unavoidable, keep it in an explicitly non-release path and make sure `DO_NOT_RELEASE` excludes it.

## Evidence

- Export-only or unit-test-only evidence is insufficient for model-facing behavior.
- For model changes, expect checkpoint export, TensorRT engine build, runtime inference, and deterministic reference comparison when applicable.
- For CI failures, require a root cause: change regression, known unrelated breakage, or infrastructure issue with evidence.

## Simplicity

- Prefer simple, scoped changes.
- Flag unrelated rebase artifacts, random platform/version churn, speculative future-generalization, duplicate source-of-truth tables, legacy compatibility paths that no producer still needs, and unreachable defensive code.
- Flag redundant transformations that wrap a single-use expression in a pass-through helper, add an abstraction with one caller and no ownership boundary, move code only to rename it, or change equivalent control flow without a correctness, testability, or measured performance reason.
- Reuse canonical helpers and predicates instead of re-deriving stringly-typed checks or duplicate validation logic.

## Comments

- Source comments should be precise and sparse. They should explain non-obvious invariants, ownership/lifetime constraints, math or precision assumptions, platform constraints, or surprising control flow.
- Flag AI-style comments that narrate what each statement does, repeat names or types already visible in code, restate obvious control flow, or add long prose without changing maintainability.
- Flag change narration: comments that describe the change relative to previous code, such as "replaces X" or "now uses Y". That content belongs in the commit message or PR description.
- Do not ask for more comments by default. Request a comment only when it would clarify a real invariant, contract, precision choice, concurrency behavior, or hardware/platform constraint.
- Greptile review comments should also be concise: one actionable finding per comment, no repeated versions of the same issue, and no generic style lecture when a short concrete fix is enough.

## CMake and Build Graph

- Review CMake changes as build-graph changes, not text-only edits.
- Flag accidental target/link/interface property regressions, global include or link-directory pollution, hidden dependency or version changes, missing source/header/test/install registration, duplicated source lists, platform guard regressions, non-hermetic absolute paths or environment assumptions, and option default changes without docs or validation.
- New kernels, plugins, builders, runtimes, and examples should wire sources, generated artifacts, dependencies, and tests through the smallest owning target instead of broad global settings.

## Public Inputs

- Validate public inputs early and fail with clear errors.
- If a mode is detected from files or config, verify the complete required artifact set before routing into runtime code.
- Prefer existing validators over ad-hoc partial detection.
