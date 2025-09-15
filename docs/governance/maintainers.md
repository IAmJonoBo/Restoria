# Maintainers

This document outlines the maintainer structure, responsibilities, and processes for GFPGAN.

## Current Maintainers

### Lead Maintainer

- **@IAmJonoBo** - Project lead, releases, infrastructure

### Core Maintainers

- **@IAmJonoBo** - General maintenance, code review, documentation

### Area Maintainers

| Area | Maintainer | Responsibilities |
|------|------------|------------------|
| **Core Engine** | @IAmJonoBo | Model architectures, inference pipeline |
| **Documentation** | @IAmJonoBo | Docs site, guides, API documentation |
| **CI/CD** | @IAmJonoBo | Build, test, and deployment automation |
| **Web Interface** | @IAmJonoBo | Gradio app, API server |

## Responsibilities

### Lead Maintainer

- **Release management**: Planning, coordination, and execution
- **Project direction**: Technical roadmap and architecture decisions
- **Community management**: Issue triage, contributor onboarding
- **Security**: Vulnerability response and security policy enforcement

### Core Maintainers

- **Code review**: Review and approve pull requests
- **Issue triage**: Label, prioritize, and route issues
- **Documentation**: Maintain and improve project documentation
- **Testing**: Ensure comprehensive test coverage

### Area Maintainers

- **Domain expertise**: Deep knowledge of specific components
- **Feature development**: Lead development in their area
- **Code review**: Review PRs affecting their domain
- **Documentation**: Maintain area-specific documentation

## Processes

### Issue Triage

Issues are triaged using these labels:

#### Priority Labels
- `priority/critical` - Security issues, data loss, crashes
- `priority/high` - Significant functionality problems
- `priority/medium` - General bugs and improvements
- `priority/low` - Minor enhancements, cleanup

#### Type Labels
- `type/bug` - Confirmed bugs
- `type/feature` - New feature requests
- `type/enhancement` - Improvements to existing features
- `type/docs` - Documentation improvements
- `type/question` - Support questions

#### Area Labels
- `area/core` - Core inference engine
- `area/models` - Model architectures and weights
- `area/cli` - Command-line interface
- `area/web` - Web interface and API
- `area/docs` - Documentation
- `area/ci` - Continuous integration

#### Status Labels
- `status/needs-info` - Waiting for more information
- `status/needs-reproduction` - Waiting for reproduction steps
- `status/blocked` - Blocked by external dependencies
- `status/wontfix` - Will not be fixed (with explanation)

### Pull Request Review

All PRs require:

1. **Automated checks**: All CI checks must pass
2. **Code review**: At least one maintainer approval
3. **Area review**: Area maintainer approval for specialized changes
4. **Documentation**: Updated docs for user-facing changes

### Review Guidelines

Reviewers should check for:

- **Functionality**: Does the change work as intended?
- **Code quality**: Is the code readable and maintainable?
- **Testing**: Are there appropriate tests?
- **Documentation**: Are user-facing changes documented?
- **Compatibility**: Does it maintain backward compatibility?
- **Performance**: Does it impact performance significantly?

### Release Process

Releases follow this process:

1. **Version planning**: Determine scope and version number
2. **Feature freeze**: No new features for patch releases
3. **Testing**: Run full test suite and manual testing
4. **Documentation**: Update changelog and version docs
5. **Release creation**: Tag and create GitHub release
6. **Distribution**: Deploy to PyPI and update docs site
7. **Announcement**: Communicate release to community

### Becoming a Maintainer

Path to maintainership:

1. **Contribution history**: Consistent, quality contributions
2. **Community involvement**: Help with issues and reviews
3. **Domain expertise**: Deep knowledge of a specific area
4. **Nomination**: Current maintainer nomination
5. **Consensus**: Agreement from lead maintainer

### Maintainer Emeritus

Former maintainers who have stepped back but contributed significantly:

- Recognition in project documentation
- Optional advisory role for major decisions
- Credit in release notes and acknowledgments

## Decision Making

### Consensus Model

- **Minor changes**: Any maintainer can approve
- **Major changes**: Require lead maintainer approval
- **Breaking changes**: Require consensus from core maintainers
- **Architecture changes**: Require community discussion

### Conflict Resolution

1. **Discussion**: Open dialogue in issues or discussions
2. **Technical review**: Evaluate technical merits
3. **Community input**: Gather broader community feedback
4. **Final decision**: Lead maintainer makes final call if needed

## Communication

### Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code changes and reviews
- **Email**: Security issues and sensitive matters

### Response Times

Target response times for maintainers:

- **Security issues**: 48 hours
- **Critical bugs**: 72 hours
- **General issues**: 1 week
- **Pull requests**: 1 week
- **Feature requests**: 2 weeks

### Meetings

- **Monthly sync**: Core maintainer coordination (as needed)
- **Release planning**: Before major releases
- **Ad-hoc**: For urgent issues or major decisions

## Support

### Community Support

- **Documentation**: Comprehensive guides and API docs
- **Examples**: Working code examples and tutorials
- **FAQ**: Common questions and solutions
- **Discussions**: Community Q&A forum

### Professional Support

For organizations requiring professional support:

- **Consulting**: Custom integration and optimization
- **Training**: Team training and best practices
- **SLA**: Service level agreements for critical applications

Contact: support@gfpgan.ai

---

**Want to become a maintainer?** Start by contributing and engaging with the community. See our [contributing guide](contributing.md) for details.
