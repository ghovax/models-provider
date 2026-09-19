# Publishing

Models Provider must be published before packages that depend on it, including Codebind.

## One-time PyPI setup

Create a pending trusted publisher for the `models-provider` project with these exact values:

- PyPI project: `models-provider`
- GitHub owner: `ghovax`
- GitHub repository: `models-provider`
- Workflow: `publish.yml`
- Environment: `pypi`

Create the `pypi` environment in the GitHub repository. No long-lived PyPI token is required.

## Release

From a clean `main` checkout:

```console
uv version 0.1.0
uv lock --check
uv build --no-sources
git tag -a v0.1.0 -m v0.1.0
git push origin v0.1.0
```

The tag runs `.github/workflows/publish.yml`, smoke-tests both distributions, generates attestations, and publishes through PyPI Trusted Publishing.
