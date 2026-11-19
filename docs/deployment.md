# Run and deploy

## Local development

Use Python 3.14, the version used to generate the checked-in results.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate` instead.
Streamlit prints the local URL, usually http://localhost:8501.
Training is cached in memory after the first load; cold starts take longer.
The app uses bundled CSV data and requires no API keys or uploaded model files.

## Validation

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest -q
python evaluate.py
```

The evaluation command rebuilds CSV results and `reports/summary.json`, including
package versions, the dataset checksum, splits, and measured performance.
The notebooks import the same pipelines used by the frontend.

## Streamlit Community Cloud

1. Sign in at https://share.streamlit.io with access to this GitHub repository.
2. Create an app, select the repository and `main` branch, and set the entrypoint to `app.py`.
3. Select Python 3.14 in advanced settings if offered by the host.
4. Deploy and review the build logs. Add the resulting app URL to the repository About section.

The code is prepared for deployment; a repository push alone does not create a hosted app.
Host runtime options can change. If Python 3.14 is unavailable, validate the dependency set
on a supported Python version before deploying and regenerate the environment metadata.
No hosting account, paid resources, or cloud secrets are configured by this repository.

Official guidance: [app deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)
and [dependencies](https://docs.streamlit.io/deploy/concepts/dependencies).

## Optional GitHub Actions

The test workflow is supplied as `docs/ci-template.yml` and is not active. To enable
it, add that file at `.github/workflows/checks.yml` using the GitHub editor or a
credential with permission to write workflows. The app itself does not need this
permission. Local tests and evaluation work independently of GitHub Actions.
