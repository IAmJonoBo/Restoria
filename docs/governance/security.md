# Security Policy

## Reporting Security Vulnerabilities

We take the security of GFPGAN seriously. If you discover a security vulnerability, please follow these guidelines for responsible disclosure.

### Private Reporting

For security issues, please **do not** create public GitHub issues. Instead, report vulnerabilities privately:

1. **Email**: Send details to security@gfpgan.ai
2. **GitHub**: Use [private vulnerability reporting](https://github.com/IAmJonoBo/GFPGAN/security/advisories/new)

### What to Include

When reporting a vulnerability, please provide:

- **Description**: Clear explanation of the issue
- **Impact**: What an attacker could accomplish
- **Reproduction**: Step-by-step instructions to reproduce
- **Environment**: Operating system, Python version, GFPGAN version
- **Proof of concept**: Code or screenshots (if applicable)

### Our Response Process

1. **Acknowledgment**: We'll confirm receipt within 48 hours
2. **Assessment**: Initial assessment within 5 business days
3. **Updates**: Regular status updates every 5 business days
4. **Resolution**: Coordinated disclosure once fixed

### Supported Versions

We provide security updates for:

| Version | Supported          |
| ------- | ------------------ |
| 1.4.x   | ✅ Full support    |
| 1.3.x   | ✅ Security fixes  |
| < 1.3   | ❌ No support     |

### Security Best Practices

When using GFPGAN:

#### Input Validation

- **Validate file types**: Only process trusted image formats
- **Size limits**: Implement reasonable file size restrictions
- **Sanitize paths**: Validate input/output file paths
- **Network isolation**: Run processing in isolated environments

```python
# Example: Safe file handling
def safe_process_image(file_path):
    # Validate file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    if not any(file_path.lower().endswith(ext) for ext in allowed_extensions):
        raise ValueError("Unsupported file format")

    # Validate file size (e.g., max 50MB)
    if os.path.getsize(file_path) > 50 * 1024 * 1024:
        raise ValueError("File too large")

    # Process with GFPGAN
    return gfpgan_infer(file_path)
```

#### API Security

- **Rate limiting**: Implement request rate limits
- **Authentication**: Secure API endpoints appropriately
- **Input sanitization**: Validate all API inputs
- **Output filtering**: Don't expose internal paths or errors

#### Model Security

- **Verify checksums**: Validate model file integrity
- **Trusted sources**: Only download models from official sources
- **Isolation**: Run inference in sandboxed environments
- **Resource limits**: Set memory and computation limits

### Known Security Considerations

#### Model Files

- GFPGAN uses PyTorch `.pth` model files
- These files can contain arbitrary Python code
- Only use models from trusted sources
- Verify file checksums when available

#### Dependencies

- Regular dependency updates via Dependabot
- Security scanning with CodeQL
- Vulnerability monitoring for third-party packages

#### Processing Risks

- **Memory exhaustion**: Large images can cause OOM
- **Path traversal**: Malicious file paths could access system files
- **Model poisoning**: Malicious models could execute arbitrary code

### Security Features

#### Built-in Protections

- **Input validation**: File format and size checking
- **Error handling**: Graceful failure without information leakage
- **Resource management**: Memory and GPU memory cleanup
- **Dependency pinning**: Locked dependency versions for reproducibility

#### Optional Security Enhancements

```bash
# Run in restricted container
docker run --rm -it --security-opt=no-new-privileges \
  --cap-drop=ALL --user=1000:1000 gfpgan:latest

# Process with resource limits
timeout 300 gfpgan-infer --input photo.jpg --device cpu
```

### Disclosure Timeline

- **Day 0**: Vulnerability reported privately
- **Day 2**: Acknowledgment sent to reporter
- **Day 7**: Initial assessment and triage
- **Day 14-30**: Fix development and testing
- **Day 30-60**: Coordinated disclosure and release
- **Day 60+**: Public disclosure (if fix is available)

### Security Updates

Security updates are released as:

1. **Patch versions**: Critical security fixes (e.g., 1.4.1 → 1.4.2)
2. **GitHub Security Advisories**: Public disclosure with details
3. **CVE assignments**: For significant vulnerabilities
4. **Release notes**: Clear security fix descriptions

### Contact Information

- **Security Email**: security@gfpgan.ai
- **PGP Key**: Available at https://gfpgan.ai/.well-known/pgp-key.txt
- **GitHub Security**: https://github.com/IAmJonoBo/GFPGAN/security

### Acknowledgments

We appreciate responsible security researchers and will acknowledge contributions in:

- Security advisories
- Release notes
- Hall of Fame (with permission)

---

**Questions about security?** Contact us at security@gfpgan.ai or review our [contributing guidelines](contributing.md).
