import face_alignment
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, device='cuda')

input = io.imread('figures/aflw-test.jpg')
preds = fa.get_landmarks(input)

# Plot the image and landmarks
plt.imshow(input)
for pred in preds:
    plt.scatter(pred[:, 0], pred[:, 1], marker='o', color='b', s=2)
plt.axis('off')
plt.show()
