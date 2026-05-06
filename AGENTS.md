# Symmetric Learning

This is a python package holding utility functions and torch modules for symmetry constrained machine learning. The README should clear the structure of the project, which holds a Sphinx docs dir, and strcutures the python package into submodules: `nn`, `linalg`, `stats`, `utils`, and `models`. Each submodule has its corresponding test file in the /test directory.

## Design principles

1. As a design principle, users are **not** expected to reason about or manually handle cache invalidation, or fast-inference vs slow-training internals.
   - `eModule` behavior should work out-of-the-box across training/validation switches.
   - Saving and loading `eModule` checkpoints should preserve correct behavior without user-side cache-management workarounds.

## Coding contract (MUST)

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

Prefer inline code by default. Only abstract after the same logic is reused in multiple real call sites.

- No features beyond what was asked.
- No abstractions for single-use code.
- Do not introduce helper functions, builder functions, wrapper methods, or utility classes that are used only once.
- Do not extract helpers only to reduce visual duplication or “organize” a constructor.
- Bad example: a single-use helper such as `build_dense_norm(...)` used in one module once. Inline that logic where it is needed.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- No duplicated error handling in class/function hierarchies. If error is checked by a third-party function, don't re-check it.
- If you write 200 lines and it could be 50, rewrite it.
- Functions used by a single class should be instance/static methods of that class not loose module-level functions.
- Preffer flat functional programming hierarchies. Small (possibly long) functions with inline comments defining sections.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## Test contract (MUST)

If you modify the codebase, you must run tests to validate that your changes do not break existing functionality. Follow these guidelines when running tests:

1. Each function/module/model should have a **single** dedicated test function that validates its utility/behavior.
   - When adding new tests, follow this pattern: one test function per function/module/model.
2. When testing changes, run **module-specific tests first**.
   - For example, if a change is made to a function in `symm_learning.linalg.irrep_radii`, run the the corresponding test function: `pytest -q test/test_linalg.py::test_irrep_radii` first.
   - Only if user asks run the entire test suite, run all tests: `pytest -q test/`.
3. If you update documentation you dont need to run tests, but if you update docstrings you must run tests to validate that the docstring formatting.

## Docstring instructions

We construct the documentation using Sphinx and the autodoc extension, therefore:
    - Read the notation convetion in `.agent/math_notation.md` before writing docstrings for any function/module/model. Follow this convention in all docstrings.
    - Read `.agent/docstring_guidelines.md` before editing docstrings. In particular, preserve the existing local narrative and notation style unless the user explicitly asks for a broader rewrite.
    - Format all docstring math using reStructuredText (reST) syntax.
    - For structural changes of the UI or new functionalities read first the documentation of the Sphinx theme used in this project.
    - If applicable function/class docstrings should include a word description of the linear algebra operation being performed, and a latex equation describing the operation. See as an example `symm_learning.linalg.lstsq` for reference.
    - Use the style of docstring formatting presented in the `symm_learning.stats` and `symm_learning.linalg` submodules as a guide. We strive for mathematical clarity and precision in our documentation.
    - Similar to pytorch documentation, if shapes are not specificed for the input output parameters clearly, add a docstring section called "Shape" after the "Returns" section to specify the expected shapes of inputs and outputs.
    - Define mathematical objects once and refer back to them. When several definitions are tightly coupled, prefer one aligned equation over several fragmented ones.
    - When a function returns structured metadata, prefer explicit typing and list the returned fields individually in the docstring.
    - Use explicit Sphinx cross-references to examples and reference pages instead of placeholder mentions.
    - Never delete docstrings, update or expand them instead. Ask for confirmation before deleting any docstring content.

## Commit message guidelines

Generate commit messages using this exact format:

v<major>.<minor>.<patch>-<type>(<optional scope>): <short imperative summary>

- <codebase change 1>
- <codebase change ...>
- <Docs/Test change 1>
- <Version bump >

## Rules

- Always include the version in the header.
- Use lowercase for `<type>` and `<scope>`.
- Write the summary in imperative mood.
- Keep the first line concise and specific with important codebase changes.
- Add 2 to 5 bullets only when they add useful detail.
- Mention tests, docs, refactors, and version-file updates when relevant.
- Prefer these types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci`, `chore`.

## Version selection rules

- Version is exposed in `pyproject.toml`
- **MAJOR**: incompatible API changes, breaking public behavior changes, removed functionality, or changes requiring users to modify existing code or workflows.
- **MINOR**: backward-compatible new features, new public API additions, or new capabilities that do not break existing usage.
- **PATCH**: backward-compatible bug fixes, performance improvements, internal refactors without public behavior changes, documentation-only changes, test-only changes, or maintenance work.

## Pre-1.0 policy

- For versions `0.y.z`, still include the full version in the title.
- For pre-1.0 releases, use this policy:
  - breaking changes or new features: increment **MINOR** and reset **PATCH** to `0`
  - bug fixes, refactors, perf improvements, docs, tests, build, ci, and chore changes: increment **PATCH**
- Do not increment **MAJOR** above `0` unless the diff clearly indicates the project is moving to a stable `1.0.0` release.

## Type guidance

- `feat` = new user-facing or API-facing functionality
- `fix` = bug fix
- `perf` = performance improvement without intended feature change
- `refactor` = code restructuring without intended behavior change
- `docs` = documentation only
- `test` = tests only
- `build` = build-system or dependency packaging changes
- `ci` = CI/CD configuration changes
- `chore` = maintenance tasks not affecting product behavior

## Examples

- `v0.4.7-fix(stats): correct covariance computation and improve projection efficiency`
- `v0.4.8-perf(linearizer): reduce allocations in projection pipeline`
- `v1.0.0-feat(api): finalize stable public interface`
- `v2.0.0-fix(core): align implementation with the new breaking API contract`
