# TECFIDERA study
This folder contains the data loaders and preprocessing scripts for parsing the TECFIDERA data.


## Prediction
The masks are not provided and need to be pre-computed using [compute_masks.py](compute_masks.py).
These masks should be passed to the `--masks` parameter of [predict_test.py](predict_test.py).

