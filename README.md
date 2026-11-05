# MovieLens 100K: EDA and reproducible baselines

Portfolio data-analysis project by Rozstka. Two executable notebooks, reusable Python analysis,
verified dataset download, chronological rating-prediction evaluation and regression tests.
This is a recommender/data-science project, not an LLM application.

## Run (Python 3.12+)
```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.lock
python scripts/download_data.py
pytest -q
python scripts/evaluate.py
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.executed.ipynb --output-dir reports
jupyter nbconvert --to notebook --execute notebooks/02_recommender_baseline.ipynb --output 02_recommender.executed.ipynb --output-dir reports
```
Use Jupyter to open the notebooks interactively if preferred. Paths work from the repository root or notebooks directory.

## Method and limitations
- Pearson item similarity uses **pairwise common viewers**, not total rating count; constant vectors and the target itself are excluded.
- Recommendation examples exclude already rated movies. These exploratory examples use the full dataset and do not claim test-set ranking performance.
- Predictive evaluation uses an 80% global chronological boundary (ties kept together), a training-only global-mean baseline and shrinkage item means (prior weight 10). Unseen items fall back to the training mean.
- RMSE and cold-item coverage are saved in `reports/metrics.json`. No hyperparameters are tuned on the test set. A ranking benchmark and user-based model are future work.
- Historical explicit ratings have selection and popularity bias; results do not measure online usefulness or business uplift.

## Data provenance
[MovieLens 100K / GroupLens](https://grouplens.org/datasets/movielens/100k/), released 1998.
Download script pins the publisher's MD5 and records the downloaded archive's SHA-256. It extracts only three known files.
Data are not committed or redistributed. Read `data/README` for the dataset's usage and attribution terms.
MD5 checks accidental changes; it is not a cryptographic authenticity guarantee.

## What changed
Added dependency lock, tests, verified download, reusable analysis and actual evaluation; removed stale notebook outputs.
Fixed the previous recommender's incorrect common-viewer count and a blanket `dropna()` that discarded films with missing metadata.
