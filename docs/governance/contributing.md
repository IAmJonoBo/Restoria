# Contributing to GFPGAN

We welcome contributions to GFPGAN! This guide will help you get started with developing and contributing to the project.

## Getting started

### Development environment setup

1. **Fork and clone the repository**

   ```bash
   git clone https://github.com/YOUR_USERNAME/Restoria.git
   cd Restoria
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**

   ```bash
   pip install -e ".[dev,metrics,web]"
   ```

   This installs:
   - **dev**: Linting, formatting, and testing tools
   - **metrics**: Quality evaluation dependencies
   - **web**: Web interface dependencies

4. **Set up pre-commit hooks**

   ```bash
   pre-commit install
   ```

### Development workflow

1. **Create a feature branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**

   - Write clear, well-documented code
   - Follow existing code style and patterns
   - Add tests for new functionality

3. **Test your changes**

   ```bash
   # Run linting and formatting
   nox -s lint

   # Run tests
   make test            # Light + default suites
   nox -s tests_full    # Full marker-aware run
   ```

4. **Commit and push**

   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

5. **Create a pull request**

   - Use a clear, descriptive title
   - Explain what your changes do and why
   - Reference any related issues
   - Ensure all checks pass

## Code style and standards

### Python code style

We use automated formatting and linting:

- **Black**: Code formatting
- **Ruff**: Fast linting and import sorting
- **Type hints**: Required for public APIs

```bash
# Format code
black .

# Check linting
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

### Commit message format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(cli): add dry-run mode for batch processing
fix(web): handle missing face detection gracefully
docs(guides): update hardware requirements
```

### Documentation

All user-facing features should include documentation.  Follow the style guide in
[`docs/governance/docs-conventions.md`](../governance/docs-conventions.md):

- **API functions**: Docstrings with examples
- **CLI commands**: Help text and guide updates
- **New features**: Usage guides and examples
- **Release-facing changes**: Update the [release playbook](../governance/release-playbook.md) if the process evolves

## Testing

### Test structure

```
tests/                 # Full test suite
├── test_gfpgan_arch.py
├── test_models.py
└── ...

tests_light/           # Quick tests for CI
├── test_basic.py
└── ...
```

### Running tests

```bash
# Quick tests (recommended for development)
pytest tests_light/ -v

# Full test suite (for comprehensive validation)
pytest tests/ -m "not gpu_required and not ort_required" -v

# Test specific functionality
pytest tests/test_gfpgan_model.py -v

# Test with coverage
pytest tests_light/ --cov=gfpgan --cov-report=html
```

### Writing tests

- Test both success and failure cases
- Use clear, descriptive test names
- Include edge cases and boundary conditions
- Mock external dependencies when needed

Example:
```python
def test_gfpgan_infer_with_valid_image():
    """Test that GFPGAN inference works with valid input image."""
    # Test implementation
    pass

def test_gfpgan_infer_with_invalid_format():
    """Test that GFPGAN handles invalid image formats gracefully."""
    # Test implementation
    pass
```

## Pull request checklist

Before submitting a pull request, ensure:

- [ ] **Code quality**
  - [ ] Code follows style guidelines (Black + Ruff)
  - [ ] All tests pass (`pytest tests_light/`)
  - [ ] No linting errors (`ruff check .`)

- [ ] **Documentation**
  - [ ] Docstrings added for new functions/classes
  - [ ] User guides updated if needed
  - [ ] CHANGELOG.md updated for user-facing changes

- [ ] **Testing**
  - [ ] New tests added for new functionality
  - [ ] Existing tests still pass
  - [ ] Edge cases considered

- [ ] **Compatibility**
  - [ ] Changes don't break existing API
  - [ ] Backward compatibility maintained
  - [ ] Cross-platform compatibility verified

## Development tasks

### Running the documentation site locally

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Serve docs locally
mkdocs serve

# Build static site
mkdocs build
```

### Building and testing packages

```bash
# Build wheel
python -m build

# Test installation
pip install dist/gfpgan-*.whl
```

### Benchmarking changes

```bash
# Run benchmark suite
python bench/run_bench.py --input inputs/samples/ --output bench_results/

# Compare with baseline
python bench/compare_results.py --baseline bench_baseline.json --current bench_results.json
```

## Release process

1. **Update version numbers**
   - `VERSION` file
   - `pyproject.toml`
   - `gfpgan/version.py`

2. **Update CHANGELOG.md**
   - Move items from "Unreleased" to new version section
   - Follow [Keep a Changelog](https://keepachangelog.com/) format

3. **Create release PR**
   - Title: "Release v1.2.3"
   - Include changelog highlights
   - Ensure all tests pass

4. **Tag and release**
   ```bash
   git tag v1.2.3
   git push origin v1.2.3
   ```

5. **Deploy documentation**
   ```bash
   mike deploy --push --update-aliases 1.2.3 latest
   ```

## Getting help

- **Questions**: Open a [discussion](https://github.com/IAmJonoBo/Restoria/discussions)
- **Bugs**: Create an [issue](https://github.com/IAmJonoBo/Restoria/issues)
- **Features**: Start with a discussion, then create an issue
- **Security**: See our [security policy](security.md)

## Code of conduct

Please read and follow our [Code of Conduct](code-of-conduct.md). We're committed to providing a welcoming and inclusive environment for all contributors.

---

Thank you for contributing to GFPGAN! 🎉
