# Docstring Guidelines

Use these guidelines together with `.agent/math_notation.md` when editing docstrings.

## Scope and Preservation

- Keep edits scoped to the requested function, class, or section.
- Preserve the user-authored narrative, notation, and prose flow unless a change is explicitly requested.
- Do not normalize wording, structure, or mathematical presentation to a different house style if the current style is already intentional and coherent.

## Formalism

- Define each mathematical object once, then refer back to it instead of redefining it in multiple places.
- When several objects are tightly coupled, prefer a single aligned or longer equation over several fragmented equations.
- Use notation consistently across the whole docstring, especially for isotypic coordinates, block decompositions, and basis expansions.
- Keep mathematical statements dense but readable: compress repeated definitions, not content.

## Documentation Structure

- Returns sections should not restate full derivations already introduced above; they should reference previously defined objects.
- If a function returns a structured object, document the structure explicitly.
- Prefer concrete typed return structures when possible, and list the fields of dictionaries or typed dictionaries individually.
- When examples or concepts are documented elsewhere in the Sphinx docs, use explicit `:ref:` links instead of placeholder prose such as "see example above".

## Style Signals To Preserve

- Follow the notation and decomposition style already established in the paired tutorial or reference page.
- Prefer mathematically precise prose over generic explanatory filler.
- If the user has already refined a documentation style in nearby text, treat that as the local style to preserve.
