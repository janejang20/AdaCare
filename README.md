# AdaCare: Explainable Clinical Health Status Representation Learning via Scale-Adaptive Feature Extraction and Recalibration

This repositry utlilizes the source code for *AdaCare: Explainable Clinical Health Status Representation Learning via Scale-Adaptive Feature Extraction and Recalibration*.

## Data
The MIMIC-III dataset is a comprehensive relational database containing 26 interconnected tables linked by unique identifiers such as SUBJECT\_ID (patient), HADM\_ID (hospital admission), and ICUSTAY\_ID (ICU stay). Designed to closely represent raw hospital data while maintaining clarity, the data model balances simplicity and accuracy by organizing tables to track patient stays, cross-reference medical codes, and record detailed patient care information without unnecessary merging of distinct data sources. To create a time series dataset suitable for decompensation detection, we utilized a Python suite designed to construct benchmark datasets from MIMIC-III. This allowed us to process and transform the raw clinical data into structured time series, capturing the sequential nature of patient measurements over time. To label decompensation events, we curated the data by checking if the patient's date of death occurred within the subsequent 24-hour window. 

**Data is not provided in this repository. In order to run this experiment, you need to download and run mimic3-benchmark and save the files in `data-sample` folder.**

## Running the Model
Run `train.py` in test mode and input the data directory. For example,
$ python train.py --test_mode=1 --data_path='./data-sample/' 