# Release Helper Scripts

## OSS Release Sanitizer

Release policy lives at the repo root in `oss_release_manifest.json`, following
the same centralized-policy shape as TensorRT's `oss_components.yml`.

Use `DO_NOT_RELEASE` for whole-file, whole-directory, and glob exclusions from
the release staging tree. Use `{$edge-llm-internal-release begin}` and
`{$edge-llm-internal-release end}` for internal regions inside files that
otherwise remain public.

Expected release/mirror usage:

```bash
OSS_DIR="$(mktemp -d)"
git archive HEAD | tar -x -C "$OSS_DIR"

python3 scripts/strip_internal_release.py --root "$OSS_DIR"
```

The tool consumes `DO_NOT_RELEASE` when present, deletes manifest-listed
internal-only files, strips guarded regions, and fails if any manifest-listed
forbidden pattern remains. Run it on a clean archive or clone, not on an
arbitrary developer build directory.

For CI or local validation of the current tracked tree, run:

```bash
python3 scripts/check_oss_release_sanitizer.py
```

## Release Documentation

Build published documentation from the sanitized tree, never from the raw
source: `DO_NOT_RELEASE` entries such as
`docs/source/user_guide/examples/omni_internal.md` otherwise render into public
HTML, and internal C++ headers reach the API pages and every nav sidebar.

`docs/source/conf.py` links each page back to the commit it was generated from.
That commit must be the public GitHub one — a `git archive` export has no
repository to read, and a GitLab SHA does not resolve on github.com. Pass it
explicitly when building release docs:

```bash
(
  cd "$OSS_DIR/docs"
  doxygen
  EDGELLM_DOCS_COMMIT_SHA=<public github commit> make html
)
```

`conf.py` fails if `docs/cpp_docs/xml/index.xml` is absent, and the Makefile
does not run Doxygen itself.

The build fails rather than emitting links to a commit that does not exist, so
an unset variable outside a repository is reported instead of silently
published.
