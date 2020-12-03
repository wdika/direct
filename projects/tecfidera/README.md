# TECFIDERA study
This folder contains the data loaders and preprocessing scripts for parsing the TECFIDERA data.


## Prediction
The data should be pre-processed, converted from cfl and saved as h5 using [preprocess_data.py](preprocessing/preprocess_data.py).
The masks should be passed to the `--masks` parameter of [predict_test.py](predict_test.py).

