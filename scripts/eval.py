"""
eval.py
-------
Evaluate a saved model on the test split and generate metrics + plots.

Inputs:
  - artifacts/models/<run_name>.(joblib|pt)
  - features/<ver>/{X.npy,y.npy,meta.json}
  - data/splits/<split>.json

Outputs:
  - artifacts/results/<run_name>/metrics_test.json
  - artifacts/results/<run_name>/{roc_curve.png, pr_curve.png, confusion_matrix.png}
"""