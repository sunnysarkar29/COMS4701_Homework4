import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import numpy as np

import math

# Create figure and styling for plotting
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.set(xlabel='dimensions (m)', ylabel='log(dmax/dmin)', title='dmax/dmin vs. dimensionality')
line_styles = {0: 'ro-', 1: 'b^-', 2: 'gs-', 3: 'cv-'}

# Plot dmax/dmin ratio
# TODO: fill in valid test numbers
for idx, num_samples in enumerate([5, 10, 50, 100]):
    # TODO: Fill in a valid feature range
    feature_range = range(1, 101)
    ratios = []
    dat = []
    for num_features in feature_range:
        # TODO: Generate synthetic data using make_classification
        X, _ = make_classification(n_samples=num_samples, n_features=num_features, n_informative=1, n_redundant=0, n_clusters_per_class=1)


        # TODO: Choose random query point from X
        query_point = X[-1]

        # TODO: remove query pt from X so it isn't used in distance calculations
        X = X[:-1]

        # TODO: Calculate distances
        distances = []
        for point in X:
            dist = point - query_point
            eucluidian_dist = 0
            for coord in dist:
                eucluidian_dist += (coord * coord)
            distances.append(math.sqrt(eucluidian_dist))

        # distances = [np.linalg.norm(point - query_point) for point in X]
        ratio = np.max(distances) / np.min(distances)
        ratios.append(ratio)
        dat.append((ratio, np.max(distances), np.min(distances)))


    print([x[1] for x in dat])
    import pdb; pdb.set_trace()

    ax.plot(feature_range, np.log(ratios), line_styles[idx], label=f'N={num_samples:,}')

plt.legend()
plt.tight_layout()
plt.grid(True)

plt.show()