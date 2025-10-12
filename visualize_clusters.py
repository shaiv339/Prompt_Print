import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from feature_logic import load_data, build_feature_matrix


CSV_PATH = os.path.join(current_dir, "synthetic_prompt_biometrics_500x20.csv")
NUM_USERS = 500
#OUTPUT_PREFIX = "hybrid"
OUTPUT_PREFIX = "styledistance_only"

print("\nLoading dataset and extracting features.")
df_sample, sampled_users = load_data(CSV_PATH, NUM_USERS)
features = build_feature_matrix(df_sample)
print(f"Features ready: {features.shape}")


pca = PCA(n_components=2, random_state=42)
coords_2d = pca.fit_transform(features)
df_sample['x'] = coords_2d[:, 0]
df_sample['y'] = coords_2d[:, 1]

var_ratio = pca.explained_variance_ratio_
print(f"PCA Variance explained: {var_ratio.sum():.2%}")

def get_color_map(n_users):
    if n_users <= 10:
        return plt.cm.tab10(np.linspace(0, 1, 10))[:n_users]
    elif n_users <= 20:
        return plt.cm.tab20(np.linspace(0, 1, 20))[:n_users]
    else:
        return plt.cm.gist_rainbow(np.linspace(0, 1, n_users))

colors_list = get_color_map(NUM_USERS)
user_color_map = {user: colors_list[i] for i, user in enumerate(sampled_users)}


print("Plotting clusters...")
fig, ax = plt.subplots(figsize=(16, 12))
for user_id in sampled_users:
    data = df_sample[df_sample['user_id'] == user_id]
    ax.scatter(
        data['x'], data['y'],
        c=[user_color_map[user_id]] * len(data),
        s=100, alpha=0.7,
        edgecolors='black', linewidth=0.8,
        label=user_id
    )

ax.set_xlabel('PC1', fontsize=14, fontweight='bold')
ax.set_ylabel('PC2', fontsize=14, fontweight='bold')
ax.set_title(f'User Writing Style Clusters (new)\n{NUM_USERS} Users × 20 Prompts', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')

ax.text(
    0.02, 0.98,
    f'Variance Explained: {var_ratio.sum():.1%}\nPC1: {var_ratio[0]:.1%} | PC2: {var_ratio[1]:.1%}',
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
)

plt.tight_layout()
plt.savefig(f'{OUTPUT_PREFIX}_user_clusters.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved: {OUTPUT_PREFIX}_user_clusters.png")


print("\nComputing cluster statistics...")
stats = []
for user_id in sampled_users:
    cluster = df_sample[df_sample['user_id'] == user_id]
    spread = np.sqrt(cluster['x'].std()**2 + cluster['y'].std()**2)
    stats.append({
        'user_id': user_id,
        'num_prompts': len(cluster),
        'spread': spread,
        'mean_x': cluster['x'].mean(),
        'mean_y': cluster['y'].mean()
    })

stats_df = pd.DataFrame(stats).sort_values('spread')
stats_df.to_csv(f'{OUTPUT_PREFIX}_cluster_stats.csv', index=False)
print("Stats saved:", f'{OUTPUT_PREFIX}_cluster_stats.csv')

print("\nTightest clusters:")
print(stats_df.head(5).to_string(index=False))

print("\nLoosest clusters:")
print(stats_df.tail(5).to_string(index=False))