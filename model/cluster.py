import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION
PROCESSED_DATA_PATH = './data/processed/'
OUTPUT_PATH = './outputs/'
N_CLUSTERS_MAX = 10
RANDOM_SEED = 42

# Create output directories
import os
os.makedirs(f'{OUTPUT_PATH}figures/', exist_ok=True)
os.makedirs(f'{OUTPUT_PATH}reports/', exist_ok=True)

# LOAD PREPROCESSED DATA
X_scaled = pd.read_csv(f'{PROCESSED_DATA_PATH}scaled_features.csv', index_col=0)
customer_data = pd.read_csv(f'{PROCESSED_DATA_PATH}customer_profiles.csv')
scaler = joblib.load(f'{PROCESSED_DATA_PATH}scaler.pkl')

print(f"✓ Loaded {len(X_scaled)} customers with {len(X_scaled.columns)} features")

# ELBOW METHOD - Find optimal number of clusters
print('elbow method')
inertias = []
silhouette_scores = []
K_range = range(2, N_CLUSTERS_MAX + 1)

for k in K_range:
    print(f"Testing k={k}...", end=' ')
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=RANDOM_SEED)
    kmeans.fit(X_scaled)
    
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(sil_score)
    
    print(f"Inertia: {kmeans.inertia_:.2f}, Silhouette: {sil_score:.3f}")

# VISUALIZE ELBOW CURVE AND SILHOUETTE SCORES
print("visualizatiions ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow curve
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[0].set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
axes[0].set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(K_range)

# Silhouette scores
axes[1].plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Score by Number of Clusters', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(K_range)

plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}figures/01_elbow_silhouette.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved elbow_silhouette.png")
plt.close()

# FIND OPTIMAL K (using silhouette score)
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\n✓ Optimal number of clusters (by silhouette): {optimal_k}")
print(f"  Silhouette Score: {silhouette_scores[optimal_k-2]:.3f}")

# FINAL K-MEANS WITH OPTIMAL K
print(f"\n--- FINAL K-MEANS CLUSTERING (k={optimal_k}) ---")

kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=RANDOM_SEED)
customer_data['Cluster'] = kmeans_final.fit_predict(X_scaled)

# Calculate validation metrics
sil_score = silhouette_score(X_scaled, customer_data['Cluster'])
db_score = davies_bouldin_score(X_scaled, customer_data['Cluster'])
ch_score = calinski_harabasz_score(X_scaled, customer_data['Cluster'])

print(f"Silhouette Score: {sil_score:.3f} (range: -1 to 1, higher is better)")
print(f"Davies-Bouldin Index: {db_score:.3f} (lower is better)")
print(f"Calinski-Harabasz Index: {ch_score:.2f} (higher is better)")

# CLUSTER ANALYSIS
print(f"\n--- cluster analysis............. ---")

cluster_summary = customer_data.groupby('Cluster').agg({
    'CustomerID': 'count',
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'sum'],
    'AvgOrderValue': 'mean',
    'ItemsPerTransaction': 'mean'
}).round(2)

print("\nCluster Summary:")
print(cluster_summary)

# CREATE DETAILED CLUSTER PROFILES
print(f"\n--- DETAILED CLUSTER PROFILES ---")

for cluster_id in sorted(customer_data['Cluster'].unique()):
    cluster_customers = customer_data[customer_data['Cluster'] == cluster_id]
    
    print(f"\n{'='*60}")
    print(f"CLUSTER {cluster_id}")
    print(f"{'='*60}")
    print(f"Size: {len(cluster_customers)} customers ({len(cluster_customers)/len(customer_data)*100:.1f}%)")
    print(f"\nRFM Metrics:")
    print(f"  Recency (days):        {cluster_customers['Recency'].mean():.1f} days")
    print(f"  Frequency (txns):      {cluster_customers['Frequency'].mean():.1f} transactions")
    print(f"  Monetary (£):          £{cluster_customers['Monetary'].mean():.2f}")
    print(f"  Avg Order Value:       £{cluster_customers['AvgOrderValue'].mean():.2f}")
    print(f"  Items per Transaction: {cluster_customers['ItemsPerTransaction'].mean():.1f}")
    print(f"\nValue Metrics:")
    print(f"  Total Cluster Revenue: £{cluster_customers['Monetary'].sum():,.2f}")
    print(f"  % of Total Revenue:    {cluster_customers['Monetary'].sum() / customer_data['Monetary'].sum() * 100:.1f}%")
    print(f"  Max Customer Value:    £{cluster_customers['Monetary'].max():.2f}")
    print(f"  Min Customer Value:    £{cluster_customers['Monetary'].min():.2f}")

# VISUALIZE CLUSTERS (PCA PROJECTION)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot clusters
fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=customer_data['Cluster'], 
                     cmap='viridis', 
                     s=50, 
                     alpha=0.6, 
                     edgecolors='k', 
                     linewidth=0.5)

# Plot cluster centers
centers_pca = pca.transform(kmeans_final.cluster_centers_)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1], 
          c='red', marker='*', s=500, edgecolors='black', linewidth=2, label='Centroids')

ax.set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
ax.set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
ax.set_title(f'Customer Clusters (PCA Projection, k={optimal_k})', fontsize=14, fontweight='bold')

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Cluster', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}figures/02_cluster_pca.png', dpi=300, bbox_inches='tight')
plt.close()

# FEATURE IMPORTANCE (PCA LOADINGS)
print(f"\n--- featur importancce ---")

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings_df = pd.DataFrame(
    loadings,
    columns=['PC1', 'PC2'],
    index=X_scaled.columns
)

fig, ax = plt.subplots(figsize=(10, 6))

loadings_df.plot(kind='barh', ax=ax, width=0.8)
ax.set_xlabel('Loading Value', fontsize=12)
ax.set_title('Feature Contributions to Principal Components', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}figures/03_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# CLUSTER SIZE AND VALUE DISTRIBUTION
print(f"\n--- CLUSTER DISTRIBUTION ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Cluster sizes
cluster_sizes = customer_data['Cluster'].value_counts().sort_index()
axes[0, 0].bar(cluster_sizes.index, cluster_sizes.values, color='steelblue', edgecolor='black')
axes[0, 0].set_xlabel('Cluster', fontsize=11)
axes[0, 0].set_ylabel('Number of Customers', fontsize=11)
axes[0, 0].set_title('Cluster Sizes', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Revenue per cluster
cluster_revenue = customer_data.groupby('Cluster')['Monetary'].sum().sort_index()
axes[0, 1].bar(cluster_revenue.index, cluster_revenue.values, color='green', edgecolor='black')
axes[0, 1].set_xlabel('Cluster', fontsize=11)
axes[0, 1].set_ylabel('Total Revenue (£)', fontsize=11)
axes[0, 1].set_title('Revenue per Cluster', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Average monetary value
cluster_avg_monetary = customer_data.groupby('Cluster')['Monetary'].mean().sort_index()
axes[1, 0].bar(cluster_avg_monetary.index, cluster_avg_monetary.values, color='orange', edgecolor='black')
axes[1, 0].set_xlabel('Cluster', fontsize=11)
axes[1, 0].set_ylabel('Avg Customer Value (£)', fontsize=11)
axes[1, 0].set_title('Average Customer Value by Cluster', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Average frequency
cluster_avg_freq = customer_data.groupby('Cluster')['Frequency'].mean().sort_index()
axes[1, 1].bar(cluster_avg_freq.index, cluster_avg_freq.values, color='purple', edgecolor='black')
axes[1, 1].set_xlabel('Cluster', fontsize=11)
axes[1, 1].set_ylabel('Avg Frequency (transactions)', fontsize=11)
axes[1, 1].set_title('Average Purchase Frequency by Cluster', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}figures/04_cluster_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# Save customer segmentation
customer_data.to_csv(f'{OUTPUT_PATH}customer_segments.csv', index=False)

# Save model
joblib.dump(kmeans_final, f'{OUTPUT_PATH}models/kmeans_model.pkl')

# Save PCA model
joblib.dump(pca, f'{OUTPUT_PATH}models/pca_model.pkl')

# Save cluster summary report
with open(f'{OUTPUT_PATH}reports/cluster_analysis_report.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("CUSTOMER SEGMENTATION - CLUSTER ANALYSIS REPORT\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Number of Clusters: {optimal_k}\n")
    f.write(f"Total Customers: {len(customer_data)}\n\n")
    
    f.write("VALIDATION METRICS:\n")
    f.write(f"  Silhouette Score: {sil_score:.3f}\n")
    f.write(f"  Davies-Bouldin Index: {db_score:.3f}\n")
    f.write(f"  Calinski-Harabasz Index: {ch_score:.2f}\n\n")
    
    f.write("CLUSTER PROFILES:\n")
    f.write("-"*70 + "\n")
    
    for cluster_id in sorted(customer_data['Cluster'].unique()):
        cluster_customers = customer_data[customer_data['Cluster'] == cluster_id]
        f.write(f"\nCluster {cluster_id}:\n")
        f.write(f"  Size: {len(cluster_customers)} customers ({len(cluster_customers)/len(customer_data)*100:.1f}%)\n")
        f.write(f"  Avg Recency: {cluster_customers['Recency'].mean():.1f} days\n")
        f.write(f"  Avg Frequency: {cluster_customers['Frequency'].mean():.1f} transactions\n")
        f.write(f"  Avg Monetary: £{cluster_customers['Monetary'].mean():.2f}\n")
        f.write(f"  Total Revenue: £{cluster_customers['Monetary'].sum():,.2f}\n")

# SUMMARYYYYYYYYY
print("\n" + 'summary')
print("K-MEANS clustering completed!")
print("="*70)
print(f"\n✓ {optimal_k} customer segments identified")
print(f"✓ Silhouette Score: {sil_score:.3f}")
print(f"\nOutput files saved to: {OUTPUT_PATH}")
print(f"  - models/ (saved K-Means and PCA models)")