# Documentation Rules

- Documentation must match current CLI names, executable flags, supported model names, precisions, platform support, and limitations.
- If docs describe a model, kernel, plugin, platform, or precision as supported, require tests or explicit limitations showing the support level.
- Keep release-facing docs free of private-only identifiers, private paths, private checkpoints, and private deployment details.
- Avoid duplicating source-of-truth tables unless the change also explains how they stay synchronized.
