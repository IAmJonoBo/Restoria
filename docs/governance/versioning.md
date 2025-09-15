# Versioning Policy

GFPGAN follows [Semantic Versioning](https://semver.org/) (SemVer) for releases and maintains clear backward compatibility guarantees.

## Version Format

Versions follow the format: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes that require user action
- **MINOR**: New features that are backward compatible
- **PATCH**: Bug fixes and security updates

### Examples

- `1.4.0` → `1.4.1`: Patch release (bug fixes)
- `1.4.0` → `1.5.0`: Minor release (new features)
- `1.4.0` → `2.0.0`: Major release (breaking changes)

## Release Types

### Patch Releases (x.y.Z)

**What's included:**

- Bug fixes
- Security updates
- Documentation improvements
- Performance optimizations (without API changes)

**Backward compatibility:** ✅ Fully compatible

**Example:** `1.4.0` → `1.4.1`

```bash
# Safe to upgrade
pip install --upgrade gfpgan
```

### Minor Releases (x.Y.z)

**What's included:**

- New features
- New CLI options
- New API methods
- Model improvements
- Dependency updates (compatible)

**Backward compatibility:** ✅ Fully compatible

**Example:** `1.4.0` → `1.5.0`

```bash
# Safe to upgrade, new features available
pip install --upgrade gfpgan
```

### Major Releases (X.y.z)

**What's included:**

- Breaking API changes
- Removed deprecated features
- Incompatible dependency updates
- Major architecture changes

**Backward compatibility:** ❌ May require code changes

**Example:** `1.4.0` → `2.0.0`

```bash
# Review migration guide before upgrading
pip install --upgrade gfpgan
```

## Public API Definition

Our public API includes:

### Command Line Interface

```bash
# Stable CLI commands
gfpgan-infer --input photo.jpg --version 1.4
gfpgan-doctor
```

### Python API

```python
# Core classes and functions
from gfpgan import GFPGANer
from gfpgan.utils import restore_image

# Public methods and parameters
restorer = GFPGANer(model_path='...', upscale=2)
result = restorer.enhance(image, has_aligned=False)
```

### Configuration Files

```yaml
# Model registry format
models:
  gfpgan_v1.4:
    url: "..."
    sha256: "..."
```

## Compatibility Guarantees

### Within Major Versions

- **CLI commands**: Existing commands continue to work
- **API methods**: Public methods maintain signatures
- **Configuration**: Existing configs remain valid
- **Model files**: Compatible model formats

### Deprecation Policy

Before removing features:

1. **Deprecation warning**: Added in minor release
2. **Documentation**: Updated with alternatives
3. **Migration guide**: Provided for complex changes
4. **Removal**: In next major release (minimum 6 months)

Example deprecation:
```python
# v1.4.0 - deprecation warning
warnings.warn("restore_face() is deprecated, use enhance() instead")

# v2.0.0 - removal
# restore_face() method removed
```

## Release Schedule

### Regular Releases

- **Patch releases**: As needed for critical fixes
- **Minor releases**: Every 2-3 months
- **Major releases**: Every 12-18 months

### Security Releases

- **Critical vulnerabilities**: Within 48 hours
- **High severity**: Within 1 week
- **Medium/Low severity**: Next regular release

## Support Windows

### Active Support

| Version | Release Date | End of Support | Security Fixes |
|---------|--------------|----------------|----------------|
| 1.4.x   | 2024-Q4     | 2025-Q4        | ✅ Yes         |
| 1.3.x   | 2024-Q2     | 2024-Q4        | ✅ Yes         |

### Legacy Support

- **Bug fixes**: Current major version only
- **Security fixes**: Current + previous major version
- **New features**: Current major version only

## Pre-release Versions

### Alpha Releases

- **Format**: `1.5.0a1`, `1.5.0a2`
- **Purpose**: Early feature testing
- **Stability**: Unstable, may have breaking changes
- **Availability**: GitHub releases only

### Beta Releases

- **Format**: `1.5.0b1`, `1.5.0b2`
- **Purpose**: Feature-complete testing
- **Stability**: More stable, API frozen
- **Availability**: GitHub releases and PyPI

### Release Candidates

- **Format**: `1.5.0rc1`, `1.5.0rc2`
- **Purpose**: Final testing before release
- **Stability**: Production-ready candidate
- **Availability**: GitHub releases and PyPI

## Migration Support

### Migration Guides

For major releases, we provide:

- **Step-by-step instructions**: How to update code
- **Breaking change summaries**: What changed and why
- **Compatibility shims**: Temporary compatibility layers
- **Example migrations**: Before/after code examples

### Tools

```bash
# Check compatibility with new version
gfpgan-doctor --check-compatibility 2.0.0

# Automated migration assistance
gfpgan-migrate --from 1.4 --to 2.0 --scan ./my_project
```

## Version Information

### Runtime Version Check

```python
import gfpgan
print(gfpgan.__version__)  # "1.4.1"

# Programmatic version comparison
from packaging import version
if version.parse(gfpgan.__version__) >= version.parse("1.4.0"):
    # Use new features
    pass
```

### CLI Version Check

```bash
gfpgan-infer --version
# GFPGAN 1.4.1
```

## Documentation Versioning

### Docs Site Versioning

- **Latest**: Current stable release
- **Development**: Main branch (unreleased)
- **Historical**: Previous major versions

Access via: `https://gfpgan.ai/docs/v1.4/`

### API Documentation

- **Stable**: Generated from release tags
- **Development**: Generated from main branch
- **Legacy**: Maintained for supported versions

## Change Communication

### Changelog

All releases include detailed changelogs:

- **Added**: New features
- **Changed**: Modifications to existing features
- **Deprecated**: Features marked for removal
- **Removed**: Deleted features
- **Fixed**: Bug fixes
- **Security**: Security improvements

### Release Notes

Major releases include:

- **Upgrade guide**: Step-by-step instructions
- **Highlights**: Key new features
- **Breaking changes**: What requires code changes
- **Performance improvements**: Benchmark comparisons

### Notifications

- **GitHub releases**: Automatic notifications
- **PyPI**: Package update notifications
- **Documentation**: Version-specific announcements

---

**Questions about versioning?** See our [FAQ](../faq.md) or [contributing guide](contributing.md).
