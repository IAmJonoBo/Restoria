# Contributing to Restoria

Thank you for your interest in contributing to Restoria! This document provides
guidelines for contributing to the project.

## Quick start

1. **Set up development environment**:

   ```bash
   git clone https://github.com/IAmJonoBo/Restoria.git
   cd Restoria
   pip install -e ".[dev,metrics,web]"
   ```

2. **Run tests**:

   ```bash
   python -m pytest tests/
   ```

3. **Preview documentation**:

   ```bash
   pip install -e ".[docs]"
   mkdocs serve
   ```

## Detailed contribution guide

For comprehensive contributing guidelines, please see our
[detailed documentation](https://IAmJonoBo.github.io/Restoria/governance/contributing/).

## Quick reference

- **Code style**: Black formatter, isort imports, flake8 linting
- **Commit messages**: Conventional Commits format
- **Pull requests**: Use the provided template
- **Issues**: Use appropriate issue templates

## Getting help

- **Documentation**: [https://IAmJonoBo.github.io/Restoria/](https://IAmJonoBo.github.io/Restoria/)
- **Discussions**: [GitHub Discussions](https://github.com/IAmJonoBo/Restoria/discussions)
- **Issues**: [GitHub Issues](https://github.com/IAmJonoBo/Restoria/issues)

## Code of conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.
