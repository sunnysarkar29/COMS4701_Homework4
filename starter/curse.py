import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import numpy as np

import random

# Create figure and styling for plotting
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.set(xlabel='dimensions (m)', ylabel='log(dmax/dmin)', title='dmax/dmin vs. dimensionality')
line_styles = {0: 'ro-', 1: 'b^-', 2: 'gs-', 3: 'cv-'}

# Plot dmax/dmin ratio
# TODO: fill in valid test numbers
for idx, num_samples in enumerate([50, 100, 500, 1000]):
    # TODO: Fill in a valid feature range
    feature_range = range(1, 101)
    ratios = []
    for num_features in feature_range:
        # TODO: Generate synthetic data using make_classification
        X, _ = make_classification(n_samples=num_samples, n_features=num_features, random_state=1)

        # TODO: Choose random query point from X
        query_idx = random.randint(0, num_samples-1)
        query_point = X[query_idx]

        # TODO: remove query pt from X so it isn't used in distance calculations

        # TODO: Calculate distances
        distances = None
        ratio = np.max(distances) / np.min(distances)
        ratios.append(ratio)

    ax.plot(feature_range, np.log(ratios), line_styles[idx], label=f'N={num_samples:,}')

plt.legend()
plt.tight_layout()
plt.grid(True)