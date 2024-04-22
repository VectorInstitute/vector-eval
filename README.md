# Vector-Eval

----------------------------------------------------------------------------------------

[![code checks](https://github.com/VectorInstitute/aieng-template/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/aieng-template/actions/workflows/code_checks.yml)
[![integration tests](https://github.com/VectorInstitute/aieng-template/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/VectorInstitute/aieng-template/actions/workflows/integration_tests.yml)
[![docs](https://github.com/VectorInstitute/aieng-template/actions/workflows/docs_deploy.yml/badge.svg)](https://github.com/VectorInstitute/aieng-template/actions/workflows/docs_deploy.yml)
[![codecov](https://codecov.io/gh/VectorInstitute/aieng-template/branch/main/graph/badge.svg)](https://codecov.io/gh/VectorInstitute/aieng-template)
[![license](https://img.shields.io/github/license/VectorInstitute/aieng-template.svg)](https://github.com/VectorInstitute/aieng-template/blob/main/LICENSE)

## Description

### Overview

The Vector RAG Evaluation framework is designed to be an intuitive and flexible tool for benchmarking the performance of RAG systems. The framework exposes an Evaluator that is configured using three components: Systems, Tasks, and Metrics.

- **Systems** encapsulate a RAG system. Systems must adhere to a common interface but can be implemented by users with arbitrary complexity. Several simple baseline systems are implemented within the framework.
- **Tasks** represent RAG datasets (inspired by the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) implementation). A Task is composed of a set of Documents and a set of Task Instances for evaluation.
- **Metrics** measure various aspects of the RAG systems, including accuracy, relevance, groundedness, and hallucination detection. Metrics can be user-defined or imported from existing frameworks such as [RAGAS](https://docs.ragas.io/en/stable/), [TruLens](https://www.trulens.org/), [Rageval](https://github.com/gomate-community/rageval) and [DeepEval](https://docs.confident-ai.com/).

### Evaluation

RAG systems evaluation is a difficult task, there are many variables and hyper-parameters that can be manipulated from underlying model performance to system design. Most RAG systems (although not always) are currently developed for Q/A applications. The following elements of a RAG system can be useful for Q/A evaluation:

- **Q** - Query/Question
- **C** - Retrieved Context
- **A** - Generated Answer
- **C\*** - Ground Truth Context
- **A\*** - Ground Truth Answer

Not all of the elements will necessarily be available. Some evaluation can be performed without ground truth context (**C\***), or ground truth answers (**A\***). Evaluation without ground truth is relevant when monitoring a system deployed in production. Ultimately, this is a somewhat simplistic view of system elements. A complex system may have many elements of intermediate state that should be evaluated. For example, a re-ranking system should evaluate the context before and after re-ranking to rigorously evaluate the impact of the re-ranking model.

#### Evaluation Without Ground Truth

- Relevance between Query and Generated Answer (*relevance_query_answer*): Evaluate the relevance of the generated answer (**A**) to the original query (**Q**).
- Groundedness of Answers (*roundedness_context_answer*): Assess how well the answer (**A**) is supported by the retrieved contexts (**C**).
- Relevance between Query and Retrieved Context (*relevance_query_context*): Evaluate the relevance of the retrieved context (**C**) to the original query (**Q**).

#### Evaluation With Ground Truth

- Compare Generated and GT Answers (*answer_correctness*): Many evaluation techniques compare the generated answer (**A**) with the GT answer (**A\***).


## 🧑🏿‍💻 Developing

### Installing dependencies

Create a new env and install the required packages:
```bash
python3 -m venv env
source env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### Running evaluation

Run evaluation for TASK (for e.g. pubmedqa) using SYSTEM (for e.g. basic_rag):
```bash
python3 veval/run.py --task <TASK> --sys <SYSTEM>
```
