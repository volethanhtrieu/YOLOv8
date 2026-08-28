# Security policy

Do not report credentials in a public issue. If a W&B key, GitHub token, SSH key,
or other credential is committed or shared, revoke it immediately and replace it.

This repository must not contain:

- `.wandb.env` or real `.env` files;
- API keys or access tokens;
- `.netrc`, SSH keys, or cloud credentials;
- private absolute server paths;
- raw training data or restricted annotations.

Use GitHub secret scanning and push protection when available. Report suspected
security issues privately to the repository maintainers.
