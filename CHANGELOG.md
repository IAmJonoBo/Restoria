# Changelog

All notable changes to GFPGAN are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation restructure with task-first information architecture
- Model card and data card for responsible AI disclosures
- Governance documentation (contributing, security, maintainers, versioning)
- CODEOWNERS file for code review assignments
- GitHub issue and PR templates with detailed fields

### Changed
- Removed all fork/upstream references from documentation (except acknowledgements)
- Restructured docs with getting-started/, guides/, api/, governance/, product/, about/ directories
- Updated mkdocs.yml with improved navigation and Material theme features
- Enhanced API documentation with FastAPI auto-docs integration

### Fixed
- Documentation organization and navigation flow
- Consistent branding and messaging throughout project

## [1.0.0] - 2024-12-01

### Added
- First standalone release as independent project
- Complete documentation overhaul with professional structure
- Responsible AI framework with model and data cards
- Comprehensive governance and security policies
- Modern CI/CD pipeline with automated testing and quality checks

### Changed
- **BREAKING**: Rebranded from fork to standalone project
- Updated all documentation to reflect independent project status
- Modernized packaging with uv and PyTorch 2.x support
- Enhanced CLI with improved error handling and user experience

### Removed
- Fork-specific references and upstream dependencies (moved to acknowledgements)
- Legacy documentation structure
- Outdated CI workflows

### Security
- Added comprehensive security policy with private vulnerability reporting
- Implemented security best practices documentation
- Added automated security scanning in CI pipeline

## [v1.3.8-fork1] - 2024-09-11

### Added
- Initial fork from upstream v1.3.8
- Fork-specific branding and documentation
- Enhanced CI workflow with modern Python versions
- Improved packaging and dependency management

### Changed
- Updated README with fork-specific information
- Modified CI badge and repository URLs
- Enhanced gitignore for modern development

---

**Migration Notes:**
- This changelog begins comprehensive tracking from v1.0.0 standalone release
- For historical changes prior to independence, see git history and upstream changelog
- Breaking changes are clearly marked and migration guides provided in documentation
- Upstream: https://github.com/TencentARC/GFPGAN
- Fork: https://github.com/IAmJonoBo/GFPGAN
