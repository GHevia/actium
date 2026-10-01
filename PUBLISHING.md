# Publishing Actium

Actium is maintained at [GHevia/actium](https://github.com/GHevia/actium).
Normal branch pushes run CI. Publishing requires a version tag and a configured
PyPI trusted publisher; pushing the repository alone does not publish a package.

## 1. Configure PyPI Trusted Publishing

The included `.github/workflows/release.yml` publishes with OpenID Connect, so
no long-lived PyPI API token is stored in GitHub.

1. Create a PyPI account, enable two-factor authentication, and verify the
   account's email address.
2. In GitHub, create an environment named `pypi` under **Settings >
   Environments**. An optional required reviewer gives releases a manual gate.
3. For the first upload, configure a pending trusted publisher on PyPI. Use
   owner `GHevia`, repository `actium`, workflow filename `release.yml`, and
   environment `pypi`. If the project already exists, add the same publisher in
   that project's publishing settings.
4. Keep the workflow filename and environment name identical to the PyPI
   publisher configuration; both are identity fields.

TestPyPI can be configured first with a separate `testpypi` environment and a
copy of the publish job pointed at `repository-url:
https://test.pypi.org/legacy/`.

## 2. Build and inspect locally

From a clean checkout:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev,jdk]"
.venv/bin/python -m pip install \
  "git+https://gitlab.orekit.org/orekit/orekit-data.git@baf158744d38ec76cf94e2d396280d545b9f0ba2"
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/python -m build
.venv/bin/python -m twine check dist/*
```

## 3. Release

The version in `pyproject.toml` and `src/actium/__init__.py` must match. PyPI
versions are immutable, so increment both for every retry that follows a
successful upload.

```bash
git add pyproject.toml src/actium/__init__.py
git commit -m "Release Actium 0.3.0"
git tag -a v0.3.0 -m "Actium 0.3.0"
git push origin main
git push origin v0.3.0
```

The tag triggers the release workflow, which first requires the tag to equal
`v` plus the package version. It then runs tests, builds the wheel and source
distribution once, checks both distributions, stores them as a GitHub artifact,
and publishes that exact artifact to PyPI. After it completes, verify the public
install in a fresh environment:

```bash
python -m venv /tmp/actium-release-check
/tmp/actium-release-check/bin/python -m pip install "actium==0.3.0" "jdk4py>=21"
/tmp/actium-release-check/bin/python -c \
  "import actium; print(actium.__version__)"
```
