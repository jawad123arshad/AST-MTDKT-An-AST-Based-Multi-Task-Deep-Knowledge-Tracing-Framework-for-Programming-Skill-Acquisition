[README.md](https://github.com/user-attachments/files/29437112/README.md)
# AST-MTDKT: AST-Based Multi-Task Deep Knowledge Tracing for Programming Skill Acquisition

A Multi-Task Deep Knowledge Tracing framework that combines **Abstract Syntax Tree (AST)-based programming skill extraction** with an **LSTM-based deep learning model** to jointly predict (1) whether a student will solve their next programming task correctly, and (2) which programming skills they will demonstrate. The framework is evaluated on Python submissions from the **Project CodeNet** dataset using leakage-free, user-level 5-fold cross-validation, and is benchmarked against a **Multi-Task Bayesian Knowledge Tracing (BKT)** baseline.

---

## Overview

Traditional Deep Knowledge Tracing (DKT) models predict student performance using only binary correctness signals, ignoring the rich information embedded in the student's actual source code. This project addresses that gap by:

1. **Parsing student Python submissions with `ast`** to automatically detect which programming concepts (loops, recursion, data structures, etc.) were used — with no manual labeling required.
2. **Encoding each interaction** as a multi-hot vector capturing *which skills were used* and *whether the submission was correct*.
3. **Training a multi-task LSTM** that shares a hidden representation between a correctness-prediction head and a skill-prediction head.
4. **Evaluating rigorously** with user-level (not interaction-level) 5-fold cross-validation to prevent data leakage between train and validation sets.
5. **Comparing against a Multi-Task Bayesian Knowledge Tracing (BKT) baseline**, implemented with learnable per-skill initial/transition/guess/slip probabilities, to assess whether deep sequential modeling outperforms classical probabilistic knowledge tracing.

---

## Key Features

- **Automatic skill extraction via AST parsing** — no manual annotation of programming concepts needed.
- **Multi-task LSTM architecture** — joint correctness + skill mastery prediction with a shared representation.
- **Bayesian Knowledge Tracing baseline** — a multi-task BKT model with learnable per-skill parameters for fair comparison.
- **Leakage-free evaluation** — 5-fold cross-validation split at the *user* level, ensuring no student appears in both train and validation sets.
- **Comprehensive metrics** — Accuracy, AUC, Precision, Recall, F1-score, reported both overall and per-skill.
- **Sliding-window sequence construction** — maximizes the number of training sequences from each student's history.

---

## Skill Taxonomy

Programming skills are extracted automatically from each submission's AST. If no specific construct is detected, the submission falls back to `basic_syntax`.

| Skill          | Description                          | Example AST Triggers                          |
|----------------|---------------------------------------|-------------------------------------------------|
| `if_else`      | Conditional statements                | `ast.If`                                       |
| `loops`        | For and while loops                   | `ast.For`, `ast.While`                         |
| `functions`    | Function definitions                  | `ast.FunctionDef`                              |
| `recursion`    | Recursive function calls              | Function call matching its own enclosing `def` |
| `arrays_lists` | List operations                       | `ast.List`, `.append()`, `.pop()`, `.extend()` |
| `dictionaries` | Dictionary structures                 | `ast.Dict`                                     |
| `strings`      | String manipulation                   | `.split()`, `.strip()`, `.replace()`, `.join()`|
| `algorithms`   | Sorting / algorithmic operations      | `.sort()`                                      |
| `basic_syntax` | General syntax (fallback)             | Assigned when no other skill is detected       |

---

## Data Representation

Each student interaction is represented as a tuple:

```
(User ID, Timestamp, Correctness, Skills)
```

### Multi-Hot Encoding

For `K` skills, each interaction is encoded as a `2K`-dimensional vector:
- The **first K dimensions** flag skills that were used **correctly**.
- The **last K dimensions** flag skills that were used **incorrectly**.

### Sequence Construction

- Interactions are sorted chronologically per student.
- Sequences are built using a **sliding window** (max length = 50, stride = 25) to generate more training samples while preserving temporal order.
- Students with fewer than 20 total interactions are filtered out to ensure sufficient history for knowledge tracing.

---

## Model Architectures

### 1. Proposed Model — Multi-Task AST-DKT (`MultiTaskDKT`)

An LSTM-based sequence model with two prediction heads sharing the same hidden representation:

- **Input:** `2K`-dimensional multi-hot interaction vectors.
- **LSTM backbone:** 2 layers, hidden size 128, dropout 0.3.
- **Correctness head:** predicts the probability of solving the *next* task correctly.
- **Skill head:** predicts the *next* skill-mastery vector.
- **Loss:** `Loss = BCE(correctness) + 0.3 × BCE(skills)`

### 2. Baseline Model — Multi-Task AST-BKT (`MultiTaskBKT`)

A Bayesian Knowledge Tracing model extended for multi-task learning, used to benchmark the deep learning approach against a classical, interpretable probabilistic method:

- Learnable per-skill parameters: **initial knowledge**, **transition (learning)**, **guess**, and **slip** probabilities.
- Sequentially updates a latent per-skill mastery probability at each timestep via Bayesian posterior updates.
- An auxiliary neural skill-prediction head is added on top of the knowledge state so the baseline can be evaluated on the same multi-task setup as the proposed model.

Both models are trained and evaluated under **identical** preprocessing, skill extraction, sequence construction, and cross-validation procedures, so any performance difference reflects the knowledge-tracing methodology itself.

---

## Dataset

This project uses **[Project CodeNet](https://github.com/IBM/Project_CodeNet)**, a large-scale dataset of competitive-programming submissions released by IBM.

- Only **Python** submissions are used.
- Submission `status` is converted to a binary label: `Accepted → 1`, anything else `→ 0`.
- Only students with **≥ 20 interactions** are retained.
- A configurable cap (`MAX_PROBLEMS`) limits how many problem folders are sampled, to keep preprocessing tractable on a single machine.

> **Note:** The dataset is not included in this repository due to its size. Download it separately from the [Project CodeNet repository](https://github.com/IBM/Project_CodeNet) and update the `DATASET_PATH` variable in the notebook to point to your local copy.

---

## Evaluation

### Cross-Validation

5-fold cross-validation is performed by splitting **user IDs** (not raw interactions) into folds, guaranteeing that no student's data leaks across the train/validation boundary.

### Metrics

Reported both as overall averages across folds and per-skill:

- Accuracy
- AUC (Area Under the ROC Curve)
- Precision
- Recall
- F1-score

### Training Configuration

| Parameter                | Value |
|---------------------------|-------|
| Optimizer                 | Adam |
| Learning rate              | 0.001 |
| Weight decay               | 1e-4 |
| Hidden size                | 128 |
| LSTM layers                | 2 |
| Dropout                    | 0.30 |
| Batch size                 | 32 |
| Max epochs                 | 30 |
| Early stopping patience    | 5 (on validation AUC) |
| Max sequence length        | 50 |
| Sliding window stride      | 25 |
| Gradient clipping          | Max norm 1.0 |
| Random seed                | 42 |

---

## Project Structure

```
.
├── AST-MTDKT_An_AST-Based_Multi-Task_Deep_Knowledge_Tracing_Framework_for_Programming_Skill_Acquisition.ipynb
└── README.md
```

The notebook is organized into the following sections:

1. Background, research problem, objectives, and contributions
2. Dataset description and preprocessing
3. AST-based programming skill extraction
4. Multi-hot encoding and sequence construction
5. Proposed Multi-Task AST-DKT model, training, and evaluation (5-fold CV)
6. Multi-Task AST-BKT baseline model, training, and evaluation (5-fold CV)
7. Per-skill performance analysis for both models

---

## Requirements

```
python>=3.9
torch
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm
```

Install with:

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm
```

A CUDA-capable GPU is recommended but not required — the code automatically falls back to CPU (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`).

---

## Usage

1. **Download Project CodeNet** and note the path to its `metadata/` and `data/` folders.
2. **Open the notebook** and update the dataset path:
   ```python
   DATASET_PATH = r"path/to/Project_CodeNet"
   ```
3. **(Optional) Adjust `MAX_PROBLEMS`** to control how many problem folders are sampled, depending on your available memory/compute.
4. **Run all cells in order.** The notebook will:
   - Parse submissions and extract AST-based skills.
   - Filter active users and build sliding-window sequences.
   - Train and evaluate the Multi-Task AST-DKT model across 5 folds.
   - Train and evaluate the Multi-Task AST-BKT baseline across 5 folds.
   - Print overall and per-skill metrics for both models.
5. Best model checkpoints per fold are saved automatically.

---

## Results

Results (overall and per-skill Accuracy / AUC / Precision / Recall / F1, averaged across 5 folds) are printed at the end of each model's training section in the notebook. Paste your own run's output here once you've trained on your full dataset, for example:

```
AVERAGE PERFORMANCE (Multi-Task AST-DKT)
--------------------------------------------------------------------
Accuracy : 0.XXXX
AUC      : 0.XXXX
Precision: 0.XXXX
Recall   : 0.XXXX
F1-Score : 0.XXXX
```

---

## Future Work

- Explore transformer-based architectures (e.g., SAINT, AKT) in place of the LSTM backbone.
- Expand the skill taxonomy with finer-grained programming concepts.
- Extend AST-based skill extraction to additional programming languages beyond Python.
- Incorporate code-quality or style features alongside structural AST features.

---

## Citation

If you use this work, please cite it as:

```
@misc{astmtdkt,
  title={AST-MTDKT: An AST-Based Multi-Task Deep Knowledge Tracing Framework for Programming Skill Acquisition},
  author={<your name>},
  year={2026}
}
```

---

## License

Add your preferred license here (e.g., MIT, Apache 2.0).

---

## Acknowledgments

- [Project CodeNet](https://github.com/IBM/Project_CodeNet) — IBM's large-scale dataset of programming submissions used for evaluation.
