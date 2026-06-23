# (Modified) DecoR: Deconfounding Time Series with Robust Regression

This repository expores a direction mentioned in the original work [DecoR](https://arxiv.org/abs/2406.07005), namely a scenario, in which the confounder acts densely on the response but sparsely on the predictor. 

For more details in form of a written report, please see [report_additional_work.pdf](report_additional_work.pdf).

Please see [README_original.md](README_original.md) for the environmental setup of the original DecoR repository.

To reproduce the experiments of the "modified" DecoR, please run from project root:
```bash
python experiments/experiments_sparse_to_x.py
```