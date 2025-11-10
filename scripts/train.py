"""
train.py
---------
CLI to train a classifier on features for a given split.

Inputs:
  - features/<ver>/{X.npy,y.npy,meta.json}
  - data/splits/<split>.json

Outputs:
  - artifacts/models/<run_name>.(joblib|pt)
  - artifacts/results/<run_name>/metrics_val.json
  - artifacts/results/<run_name>/run_config.json

Flow:
  load features -> slice by split -> (optional) scale -> fit model -> val metrics -> save artifacts
"""