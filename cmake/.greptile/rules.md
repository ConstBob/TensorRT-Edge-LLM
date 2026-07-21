# CMake Rules

- Review CMake changes as build-graph changes, not text-only edits.
- Flag accidental target/link/interface property regressions, global include or link-directory pollution, hidden dependency or version changes, missing source/header/test/install registration, duplicated source lists, platform guard regressions, non-hermetic absolute paths or environment assumptions, and option default changes without docs or validation.
- New kernels, plugins, builders, runtimes, and examples should wire sources, generated artifacts, dependencies, and tests through the smallest owning target instead of broad global settings.
