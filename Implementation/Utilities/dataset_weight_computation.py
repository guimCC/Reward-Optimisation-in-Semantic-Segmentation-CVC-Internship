import numpy as np

# Class frequency of the dataset
CLS_COUNTS = [
    2036048947,
    336030285,
    1259773717,
    36211195,
    48487347,
    67771817,
    11510397,
    30522367,
    878732742,
    63964778,
    221459205,
    67202385,
    7444903,
    386502898,
    14775005,
    12995799,
    12863932,
    5445909,
    22849764,
]
# Class names of the dataset
CLS_NAMES = [
    "Road", "Sidewalk", "Building", "Wall", "Fence", "Pole", "Traffic Light", 
    "Traffic Sign", "Vegetation", "Terrain", "Sky", "Person", "Rider", "Car", 
    "Truck", "Bus", "Train", "Motorcycle", "Bicycle", "Void"
]
# Smoothing value
CLS_SMOOTH = 1

# Number of classes the dataset
N = 19

cls_counts = np.array(CLS_COUNTS)
total_counts = np.sum(cls_counts)


cls_weights = 1 / (cls_counts + CLS_SMOOTH * total_counts)
cls_weights = N * cls_weights / np.sum(cls_weights)

for class_id, class_name in zip(range(N), CLS_NAMES):
    print(f'{class_name}: Weight = {cls_weights[class_id]:.6f}')