# AGENTS instruction

## Running Commands
- In this directory, all python commands should be run using poetry, e.g. 'poetry run pytest...',  'poetry run python3 ...', 'poetry add [package-name]'.

## Coding style

- Don't explain every line with a separate comment, use comments for complex chunks of code,
  or DSL logic. Don't write docstrings for single-line or simple functions/methods.
- Use structures and code style that are already present in the codebase. Don't introduce
  another approach for new changes.
- Read `CONTRIBUTING.md` file for more details how to build the project and run tests.

## Data Model
- When working with DataLoader and other code that saves data to a file directory, read the DATA_MODEL.md file and
  ensure that the resulting directory structure is compatible with it.


## Testing
- After major code changes run the relevant portion of the test suite 'poetry run pytest ...', run mypy and fix error, and run pre-commit and fix issues.
