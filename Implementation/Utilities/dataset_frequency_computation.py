import os
import numpy as np
from PIL import Image
from tqdm import tqdm

root_annotation_dir = ''


N = 
pixel_counts = np.zeros(N, dtype=np.int64)

annotation_files = [os.path.join(root_annotation_dir, f) for f in os.listdir(root_annotation_dir) if f.endswith('.png')]

for file_path in tqdm(annotation_files, desc='Processing images'):
    img = Image.open(file_path)
    annotation_array = np.array(img)
    
    for class_id in range(N):
        pixel_counts[class_id] += np.sum(annotation_array == class_id)
        


total_pixels = pixel_counts.sum()
frequencies = pixel_counts / total_pixels

class_names = ('')

for class_id, class_name in enumerate(class_names):
    print(f'{class_name}: {frequencies[class_id]:.4f} ({pixel_counts[class_id]} pixels)')