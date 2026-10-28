# Lightweight DoS and Intrusion Detection (entropy + mutual information)

Code and result files for the CN course project report.

## Data (not included; download from the official pages)
- NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html  (KDDTrain+.txt, KDDTest+.txt)
- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset  (UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv)
Place them in a `data/` folder and set the paths at the top of `lib.py`
(the scripts expect KDDTrain+.txt, KDDTest+.txt, unsw_train.csv, unsw_test.csv).

## Run
    pip install scikit-learn xgboost scipy pandas numpy matplotlib
    python run_all.py          # experiments, JSON results into results/
    python make_results.py     # tables and figures
    python make_small_tables.py
    python fig_static.py