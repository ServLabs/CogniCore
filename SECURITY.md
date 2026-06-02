# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

1. **Do NOT** open a public GitHub issue for security vulnerabilities.
2. Use **[GitHub Security Advisories](https://github.com/ServLabs/CogniCore/security/advisories/new)** to report privately.
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (optional)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Resolution Timeline**: Depends on severity, typically 30-90 days
- **Credit**: We will credit reporters in release notes (unless you prefer anonymity)

### Scope

The following are in scope:
- CogniCore core codebase
- Official Docker images
- Documentation that could lead to security issues

Out of scope:
- Third-party dependencies (report to their maintainers)
- Social engineering attacks
- Physical security

## Security Best Practices

When deploying CogniCore:

1. **API Keys**: Never commit API keys. Use environment variables.
2. **Network**: Run behind a reverse proxy (nginx) in production.
3. **Data**: Enable encryption at rest for sensitive data.
4. **Updates**: Keep dependencies updated.
5. **Audit**: Review audit logs regularly.

## Known Security Considerations

- **PII Redaction**: Enable `core.redact` for user-facing logs
- **LLM Prompts**: Sanitize user input before including in prompts
- **Sandbox**: Code execution uses isolated sandboxes

Thank you for helping keep CogniCore secure.
