# Security Policy

## Supported Versions

Only the latest release on the `main` branch receives security fixes.
Pre-release versions and feature branches are not covered.

| Version | Supported |
|---------|-----------|
| `main` (latest) | ✅ |
| Older tags | ❌ |

## Reporting a Vulnerability

**Please do not open a public GitHub Issue for security vulnerabilities.**

### Option A — GitHub Security Advisories (preferred)

Use the [GitHub Private Security Advisory][advisory] feature:

1. Go to the repository's **Security** tab.
2. Click **"Report a vulnerability"**.
3. Fill in the description, impact, and any reproduction steps.
4. Submit — only repository maintainers can see the report.

### Option B — E-mail

Send a GPG-encrypted e-mail to **security@zoltty.ai** with the following
information:

- A clear description of the vulnerability
- Affected component(s) and version(s)
- Steps to reproduce (proof-of-concept code, if available)
- Potential impact assessment
- Your preferred disclosure timeline

Our PGP key fingerprint is published at `https://zoltty.ai/.well-known/security.txt`.

## Response Timeline

| Stage | Target |
|-------|--------|
| Initial acknowledgment | ≤ 72 hours |
| Triage and severity assessment | ≤ 7 days |
| Patch or mitigation plan shared with reporter | ≤ 14 days |
| Public disclosure (coordinated) | ≤ 90 days after report |

We follow coordinated disclosure. We will notify you before any public
announcement and give you credit in the advisory unless you prefer to remain
anonymous.

## Scope

### In scope

- `zolt/` package code (model, inference, data pipeline, tokenizer, probes)
- GitHub Actions workflows in `.github/workflows/`
- Dependencies declared in `pyproject.toml`

### Out of scope

- Issues in third-party dependencies that are already publicly disclosed
- Vulnerabilities requiring physical access to the system
- Social engineering attacks
- Denial of service via deliberate resource exhaustion during training

## Disclosure Policy

Once a fix is released, we will publish a GitHub Security Advisory with:

- CVE identifier (if assigned)
- Affected versions
- Patch version
- Credits to the reporter (unless anonymity is requested)

## Hall of Thanks

We appreciate every responsible disclosure. Contributors who report valid
security issues will be acknowledged here.

_No reports yet — be the first._

[advisory]: https://github.com/ZolttyAI/zolt/security/advisories/new
