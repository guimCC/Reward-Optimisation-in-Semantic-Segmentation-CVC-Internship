import numpy as np

# Class frequency of the dataset
CLS_COUNTS = [
    '',
]
# Class names of the dataset
CLS_NAMES = [
    '',
]
# Smoothing value
CLS_SMOOTH = 1

# Number of classes the dataset
N = 

cls_counts = np.array(CLS_COUNTS)
total_counts = np.sum(cls_counts)


cls_weights = 1 / (cls_counts + CLS_SMOOTH * total_counts)
cls_weights = N * cls_weights / np.sum(cls_weights)

for class_id, class_name in zip(range(N), CLS_NAMES):
    print(f'{class_name}: Weight = {cls_weights[class_id]:.6f}')