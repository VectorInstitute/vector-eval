# Vector-Eval

----------------------------------------------------------------------------------------

[![code checks](https://github.com/VectorInstitute/aieng-template/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/aieng-template/actions/workflows/code_checks.yml)
[![integration tests](https://github.com/VectorInstitute/aieng-template/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/VectorInstitute/aieng-template/actions/workflows/integration_tests.yml)
[![docs](https://github.com/VectorInstitute/aieng-template/actions/workflows/docs_deploy.yml/badge.svg)](https://github.com/VectorInstitute/aieng-template/actions/workflows/docs_deploy.yml)
[![codecov](https://codecov.io/gh/VectorInstitute/aieng-template/branch/main/graph/badge.svg)](https://codecov.io/gh/VectorInstitute/aieng-template)
[![license](https://img.shields.io/github/license/VectorInstitute/aieng-template.svg)](https://github.com/VectorInstitute/aieng-template/blob/main/LICENSE)

A repository for evaluating RAG systems.

## 🧑🏿‍💻 Developing

### Installing dependencies

Create a new env and install the required packages:
```bash
python3 -m venv env
source env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Run evaluation for TASK (for e.g. pubmedqa) using SYSTEM (for e.g. basic_rag):
```bash
python3 veval/cli.py --task <TASK> --sys <SYSTEM>
```
