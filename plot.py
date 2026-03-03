import matplotlib.pyplot as plt
import numpy as np

# User counts
user_counts = [500, 1000, 1500, 2000, 2500]
x = np.arange(len(user_counts))

# Methods and Colors
methods = ['Combined', 'Embedding-only', 'Stylometry-only']
colors = {
    'Combined': '#55a868',
    'Embedding-only': '#c44e52',
    'Stylometry-only': '#8172b2',
}

accuracy_data = {
    'Combined':       [0.915, 0.847, 0.8002, 0.746, 0.676],
    'Embedding-only': [0.917, 0.842, 0.794, 0.738, 0.669],
    'Stylometry-only': [0.583, 0.525, 0.496, 0.468, 0.441],
}

bar_width = 0.22  # Slightly wider bars
spacing = 0.01

plt.figure(figsize=(16, 6))

for idx, method in enumerate(methods):
    offset = (idx - 1) * (bar_width + spacing)
    plt.bar(x + offset, accuracy_data[method], width=bar_width,
            label=method, color=colors[method], edgecolor='black', alpha=0.95)

# Labels and formatting
plt.xticks(x, user_counts)
plt.xlabel('Number of Users')
plt.ylabel('Top-1 Accuracy (%)')
plt.title('Top-1 Accuracy: Combined vs Embedding vs Stylometry\n(intfloat-e5-base, 20 Prompts per User)')
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()