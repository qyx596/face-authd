# Security Policy

## Supported Versions

This project is currently pre-1.0 and evolves quickly.  
Security fixes are generally applied to the latest mainline state.

## Reporting a Vulnerability

Please do **not** open a public issue for vulnerabilities.

Until a dedicated private channel is published, report privately to the maintainer/repository owner directly.

When reporting, include:

- affected version / commit
- environment (distro, kernel, PAM stack)
- reproduction steps
- impact assessment
- proof-of-concept (if available)

## Scope Notes

This project handles biometric authentication signals and local PAM integration.  
Please report issues related to:

- authentication bypass
- privilege escalation
- insecure key handling
- sensitive data exposure (embeddings, model artifacts, logs)
- unsafe defaults in packaging or service configuration

## Response Expectations

Best effort:

- acknowledge receipt quickly
- triage and reproduce
- patch and coordinate disclosure

## Hardening Recommendations for Deployers

- keep password fallback enabled during rollout
- restrict access to `/var/lib/face-authd`
- audit PAM configuration after install/uninstall
- monitor daemon logs for repeated auth failures
