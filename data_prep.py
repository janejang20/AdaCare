import os
import pandas as pd
import numpy as np
import random

os.listdir('data')

train_listfile = pd.read_csv('data-sample/decompensation/train/listfile.csv')
test_listfile = pd.read_csv('data-sample/decompensation/test/listfile.csv')
test_size = test_listfile['stay'].str[:5].nunique()

unique_pats_train = train_listfile['stay'].str[:5].unique()
unique_pats_test = test_listfile['stay'].str[:5].unique()

print(len(unique_pats_train))
print(len(unique_pats_test))

vals = random.sample(sorted(unique_pats_train), test_size)
val_listfile = train_listfile[train_listfile['stay'].str[:5].isin(vals)]

val_listfile.to_csv('data-sample/val_listfile.csv', index=False)
train_listfile_new = train_listfile[~train_listfile['stay'].str[:5].isin(vals)]
train_listfile_new.to_csv('data-sample/train_listfile.csv', index=False)
# train_listfile_new.to_csv('data-sample/decompensation/train/listfile.csv', index=False)
test_listfile.to_csv('data-sample/test_listfile.csv', index=False)

# test_listfile['stay'].str[:5].nunique()
# print(len(unique_pats_train))

# pats_5000 = random.sample(sorted(unique_pats_train), 6000)
# pats_val = random.sample(pats_5000, 1000)
# pats_train = [i for i in pats_5000 if i not in pats_val]
# pats_test = random.sample(sorted(unique_pats_test), 1000)

# train_listfile_new = train_listfile[train_listfile['stay'].str[:5].isin(pats_train)]
# val_listfile_new = train_listfile[train_listfile['stay'].str[:5].isin(pats_val)]
# test_listfile_new = test_listfile[test_listfile['stay'].str[:5].isin(pats_test)]

# train_listfile_new.to_csv('data/train_listfile.csv', index=False)
# train_listfile_new.to_csv('data/train/listfile.csv', index=False)
# val_listfile_new.to_csv('data/val_listfile.csv', index=False)
# test_listfile_new.to_csv('data/test_listfile.csv', index=False)
# test_listfile_new.to_csv('data/test/listfile.csv', index=False)