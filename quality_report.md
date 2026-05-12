# Repository Quality Report — `ml-ids-v4`

Date: 2026-05-12

## Scope
This review covered repository structure, reproducibility, training/inference code quality, documentation consistency, dependency health, and basic operational readiness.

## Executive Summary
The project has a clear end-to-end intent (preprocess → train → Streamlit infer), but it currently has **reliability and maintainability gaps** that can break fresh setup, reduce trust in reported methodology, and increase production risk.

Top issues:
1. **Fresh setup is broken** due to missing dependency (`imblearn`) in requirements.
2. **Documentation and code are inconsistent** (docs mention one-hot/correlation/SMOTE pipeline while code uses label encoding + no SMOTE path).
3. **No automated tests or CI quality gates**.
4. **Inference robustness issues** (unknown category fallback behavior, full-file memory loading, weak input constraints).
5. **Repository hygiene gaps** (no `.gitignore`, tracked heavy artifacts/data, stale or unused artifacts).

## Findings and Improvements

### 1) Missing dependency breaks training (High)
**Evidence**
- `src/train.py` imports `from imblearn.over_sampling import SMOTE`.
- `requirements.txt` does not include `imbalanced-learn`.
- Running `python src/train.py` fails with `ModuleNotFoundError: No module named 'imblearn'`.

**Impact**
- New contributors cannot run training from a clean environment.
- Reproducibility claims are weakened.

**What to do**
- Add `imbalanced-learn` to `requirements.txt`.
- Pin core library versions (at least major/minor) for deterministic environments.
- Add a setup sanity-check command in docs (e.g., import check script).

---

### 2) Methodology drift between docs and implementation (High)
**Evidence**
- `src/train.py` header says: one-hot + correlation selection + SMOTE.
- `src/preprocessing.py` actually uses LabelEncoder + MinMaxScaler + engineered ratios.
- `apply_smote()` exists in `src/train.py` but is never called.
- Model artifacts include mixed names (`onehot_columns.joblib`, `selected_features.joblib`) that do not match active pipeline behavior.

**Impact**
- Reviewers/users cannot trust what pipeline truly produced the metrics.
- Harder debugging and future enhancement.

**What to do**
- Align docs, module docstrings, and code to one canonical pipeline.
- Remove dead paths/artifacts or re-enable them intentionally with explicit flags.
- Add a pipeline version marker saved with each trained model artifact.

---

### 3) No tests and no CI checks (High)
**Evidence**
- No test files found.
- No GitHub workflow files found.

**Impact**
- Regressions can be introduced silently.
- Security and reliability issues are likely to persist longer.

**What to do**
- Add minimal automated test set:
  - preprocessing unit tests (column normalization, unknown category handling)
  - smoke test for training entrypoint on tiny fixture data
  - app-level transform/predict smoke test
- Add CI workflow for install + tests + lint on pull requests.

---

### 4) Inference handling of unseen categories is weak (Medium)
**Evidence**
- In `transform_new`, unseen categorical values are coerced to `le.classes_[0]`.

**Impact**
- Unknown real-world values are silently mapped to an arbitrary known class.
- This can bias predictions and hide data quality issues.

**What to do**
- Introduce explicit unknown token handling for categorical features.
- Log/report unknown-value rates in app output.
- Consider encoding strategy that safely ignores/represents unknown categories.

---

### 5) Streamlit input safety and scalability concerns (Medium)
**Evidence**
- Uploaded CSV/parquet files are read fully into memory.
- No explicit file-size, row-count, or schema hard limits before expensive operations.

**Impact**
- Large uploads can cause high memory use, latency spikes, or app crashes.

**What to do**
- Add hard limits for file size and max rows.
- Validate required columns early and fail fast with actionable errors.
- Consider chunked reading or sampled preview for large files.

---

### 6) Repository hygiene and portability issues (Medium)
**Evidence**
- No `.gitignore` in repository root.
- Large generated/data/model artifacts are tracked (`data/`, `models/`, `plots/`).

**Impact**
- Repo size growth, slower clone/CI, accidental artifact drift.

**What to do**
- Add `.gitignore` for caches, temp files, and generated outputs.
- Decide artifact policy:
  - keep lightweight sample artifacts only, or
  - use release assets / model registry / external storage for heavy files.

---

### 7) Code organization and maintainability improvements (Low/Medium)
**Evidence**
- `app.py` is a large monolith containing UI, ingestion logic, and reporting logic.
- Repeated literals and mixed concerns make extension harder.

**Impact**
- Higher change risk and slower iteration.

**What to do**
- Split app into modules (`io`, `validation`, `metrics`, `ui_components`).
- Centralize constants/schema definitions.
- Add structured logging (instead of print-only traces in preprocessing/training paths).

## Prioritized Action Plan

### Quick wins (1–2 days)
1. Fix dependencies (`imbalanced-learn`) and verify clean setup.
2. Reconcile docs/docstrings with actual pipeline.
3. Add `.gitignore` and remove stale/generated files from version control policy.

### Near-term (1 week)
4. Add minimal tests (preprocessing + training smoke + inference smoke).
5. Add PR CI workflow for install/test/lint.
6. Add upload constraints and schema validation in Streamlit.

### Mid-term (2–3 weeks)
7. Refactor `app.py` into smaller modules.
8. Add model/pipeline metadata versioning and reproducibility manifest.
9. Improve unknown-category handling with explicit monitoring.

## Overall Quality Assessment
- **Current status:** Functional prototype with good project intent, but operationally fragile.
- **Production readiness:** Low (needs reproducibility, testing, and runtime hardening).
- **After recommended fixes:** Can reach stable academic/demo-quality quickly and production-candidate quality with additional engineering hardening.
