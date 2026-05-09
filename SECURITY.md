# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email the maintainers directly or use GitHub's private vulnerability reporting
3. Include a description of the vulnerability and steps to reproduce

We will acknowledge receipt within 48 hours and provide a timeline for resolution.

## Supply Chain Security (SLSA)

This project follows [SLSA](https://slsa.dev/) (Supply-chain Levels for Software Artifacts) principles:

- **Source**: All code is version-controlled in this repository
- **Build**: Releases are built via GitHub Actions with provenance attestation
- **Dependencies**: All dependencies are pinned in `pyproject.toml`
- **Provenance**: Release artifacts include SLSA provenance metadata

### Verifying Release Provenance

Releases include SLSA provenance attestations. To verify:

```bash
gh attestation verify <artifact> --repo <owner/repo>
```

## Dependency Security

- Dependencies are kept to a minimum and pinned to known-good versions
- No network calls are made during inference (fully offline operation)
- All pre-trained models are loaded from local storage
