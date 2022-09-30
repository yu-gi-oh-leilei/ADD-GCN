import numpy as np


labels_path = './val_label_vectors_coco14.npy'
labels = np.load(labels_path).astype(np.float64)
labels = (labels > 0).astype(np.float64)
print(labels.shape)
labels = np.sum(labels,axis=0)
print(labels.shape)
print(labels)



labels_path = './train_label_vectors_coco14.npy'
labels = np.load(labels_path).astype(np.float64)
labels = (labels > 0).astype(np.float64)
print(labels.shape)
labels = np.sum(labels,axis=0)
print(labels.shape)
print(labels)