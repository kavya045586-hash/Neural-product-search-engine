"""
STEP 5: EVALUATION & PERFORMANCE METRICS
Comprehensive evaluation of your recommendation system
"""

import pandas as pd
import numpy as np
import json
import os
import random
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("RECOMMENDATION SYSTEM EVALUATION")
print("=" * 70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = r"C:\Users\nagar\Downloads\parquet\processed_data"
VECTORIZED_DIR = os.path.join(DATA_DIR, "vectorized")
MAPPINGS_DIR = os.path.join(DATA_DIR, "mappings")
OUTPUT_DIR = os.path.join(VECTORIZED_DIR, "evaluation_results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# STEP 1: LOAD ALL DATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: LOADING DATA")
print("=" * 70)

# Load validation and test sets
val_df = pd.read_parquet(os.path.join(VECTORIZED_DIR, 'val_final_mini.parquet'))
test_df = pd.read_parquet(os.path.join(VECTORIZED_DIR, 'test_final_mini.parquet'))

print(f"✓ Validation set: {len(val_df):,} rows")
print(f"✓ Test set: {len(test_df):,} rows")

# Load item catalog
all_items_df = pd.read_parquet(os.path.join(VECTORIZED_DIR, 'all_items_processed_100k.parquet'))
print(f"✓ Item catalog: {len(all_items_df):,} items")

# Load embeddings
item_embeddings = np.load(os.path.join(VECTORIZED_DIR, 'item_embeddings_100k.npy'))
print(f"✓ Item embeddings: {item_embeddings.shape}")

# Load model
model = tf.keras.models.load_model(os.path.join(VECTORIZED_DIR, 'two_tower_model_100k.keras'))
print(f"✓ Model loaded: {model.name}")

# Load mappings
with open(os.path.join(MAPPINGS_DIR, 'item_mapping.json'), 'r') as f:
    item_mapping = json.load(f)

print(f"✓ Item mapping: {len(item_mapping):,} items")

# ============================================================================
# STEP 2: CREATE ITEM ID TO INDEX MAPPING
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: CREATING INDEX MAPPINGS")
print("=" * 70)

# Map item_id to its index in the embeddings array
item_id_to_index = pd.Series(all_items_df.index, index=all_items_df['item_id']).to_dict()
print(f"✓ Created item_id to index mapping: {len(item_id_to_index):,} entries")

# ============================================================================
# STEP 3: EXTRACT USER TOWER
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: EXTRACTING USER TOWER")
print("=" * 70)

try:
    user_embedding_layer = model.get_layer('user_emb')
    print(f"✓ Found user embedding layer")
except:
    print("❌ Could not find user embedding layer")
    exit(1)

# ============================================================================
# STEP 4: RANKING METRICS (HIT RATE & MRR)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: CALCULATING RANKING METRICS")
print("=" * 70)

TOP_K = 10
NUM_NEGATIVES = 50
NUM_EVAL_SAMPLES = 1000  # Evaluate on 1000 samples for speed

print(f"Configuration:")
print(f"  Top-K: {TOP_K}")
print(f"  Negative samples per positive: {NUM_NEGATIVES}")
print(f"  Evaluation samples: {NUM_EVAL_SAMPLES}")

# Get all item IDs for negative sampling
all_item_ids = list(item_id_to_index.keys())

# Metrics
hits = 0
reciprocal_ranks = []
total_evaluated = 0

print(f"\nEvaluating on validation set...")
eval_data = val_df.head(NUM_EVAL_SAMPLES)

for idx, row in eval_data.iterrows():
    # Get user embedding
    user_id = int(row['user_id'])
    user_vec = user_embedding_layer(np.array([user_id]))
    user_vec = tf.reshape(user_vec, (1, 64)).numpy()  # Assuming 64-dim
    
    # Get positive item
    pos_item_id = int(row['item_id'])
    
    if pos_item_id not in item_id_to_index:
        continue  # Skip if item not in catalog
    
    pos_idx = item_id_to_index[pos_item_id]
    pos_item_vec = item_embeddings[pos_idx].reshape(1, -1)
    
    # Get negative items
    neg_vectors = []
    while len(neg_vectors) < NUM_NEGATIVES:
        random_id = random.choice(all_item_ids)
        if random_id != pos_item_id and random_id in item_id_to_index:
            neg_idx = item_id_to_index[random_id]
            neg_vectors.append(item_embeddings[neg_idx])
    
    neg_vectors = np.array(neg_vectors)
    
    # Calculate scores
    pos_score = np.dot(user_vec, pos_item_vec.T).item()
    neg_scores = np.dot(user_vec, neg_vectors.T).flatten()
    
    # Rank them
    all_scores = np.append(neg_scores, pos_score)
    sorted_indices = np.argsort(all_scores)[::-1]
    
    # Find rank of positive item
    rank = np.where(sorted_indices == NUM_NEGATIVES)[0][0] + 1
    
    # Update metrics
    if rank <= TOP_K:
        hits += 1
    reciprocal_ranks.append(1 / rank)
    
    total_evaluated += 1
    
    if total_evaluated % 200 == 0:
        print(f"  Evaluated {total_evaluated}/{NUM_EVAL_SAMPLES} samples...")

# Calculate final metrics
hit_rate = hits / total_evaluated
mrr = np.mean(reciprocal_ranks)

print(f"\n{'='*70}")
print(f"RANKING METRICS (Validation Set)")
print(f"{'='*70}")
print(f"Hit Rate@{TOP_K}: {hit_rate:.4f} ({hit_rate*100:.2f}%)")
print(f"MRR: {mrr:.4f}")
print(f"Total Evaluated: {total_evaluated}")

# Interpretation
print(f"\n📊 Interpretation:")
if hit_rate > 0.4:
    print(f"✅ Excellent! Your model ranks relevant items in top-{TOP_K} {hit_rate*100:.1f}% of the time")
elif hit_rate > 0.3:
    print(f"✓ Good! Hit rate of {hit_rate*100:.1f}% is solid")
elif hit_rate > 0.2:
    print(f"⚠️  Fair. Hit rate of {hit_rate*100:.1f}% could be improved")
else:
    print(f"❌ Low hit rate of {hit_rate*100:.1f}%. Model needs improvement")

if mrr > 0.4:
    print(f"✅ Great MRR! Relevant items appear high in rankings")
elif mrr > 0.3:
    print(f"✓ Good MRR. Reasonable ranking quality")
else:
    print(f"⚠️  MRR could be better. Relevant items not ranked high enough")

# ============================================================================
# STEP 5: CATEGORY CLUSTERING ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: CATEGORY CLUSTERING ANALYSIS")
print("=" * 70)

print("Analyzing how well embeddings cluster by category...")

# Sample items from top categories
top_categories = all_items_df['main_category'].value_counts().head(5).index.tolist()
sampled_items = []

for cat in top_categories:
    cat_items = all_items_df[all_items_df['main_category'] == cat].head(50)
    sampled_items.append(cat_items)

sample_df = pd.concat(sampled_items)
sample_indices = sample_df.index.tolist()
sample_embeddings = item_embeddings[sample_indices]

# Calculate similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(sample_embeddings)

# Within-category vs between-category similarity
within_cat_sims = []
between_cat_sims = []

for i, row1 in sample_df.iterrows():
    for j, row2 in sample_df.iterrows():
        if i >= j:
            continue
        
        idx_i = sample_df.index.get_loc(i)
        idx_j = sample_df.index.get_loc(j)
        sim = similarity_matrix[idx_i, idx_j]
        
        if row1['main_category'] == row2['main_category']:
            within_cat_sims.append(sim)
        else:
            between_cat_sims.append(sim)

within_avg = np.mean(within_cat_sims)
between_avg = np.mean(between_cat_sims)
separation = within_avg - between_avg

print(f"\n{'='*70}")
print(f"CATEGORY CLUSTERING METRICS")
print(f"{'='*70}")
print(f"Within-category similarity: {within_avg:.4f}")
print(f"Between-category similarity: {between_avg:.4f}")
print(f"Separation score: {separation:.4f}")

if separation > 0.1:
    print(f"✅ Excellent! Strong category separation")
elif separation > 0.05:
    print(f"✓ Good category clustering")
else:
    print(f"⚠️  Weak category separation. Model may need improvement")

# ============================================================================
# STEP 6: DIVERSITY ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: RECOMMENDATION DIVERSITY ANALYSIS")
print("=" * 70)

def analyze_diversity(query, top_k=10):
    """Analyze diversity of recommendations"""
    mask = all_items_df['title'].str.contains(query, case=False, na=False)
    matches = all_items_df[mask]
    
    if len(matches) == 0:
        return None
    
    target_row = matches.iloc[0]
    target_idx = target_row.name
    target_vector = item_embeddings[target_idx].reshape(1, -1)
    
    sim_scores = np.dot(item_embeddings, target_vector.T).flatten()
    sorted_indices = np.argsort(sim_scores)[::-1]
    
    # Get top-k recommendations
    recommendations = []
    count = 0
    for idx in sorted_indices:
        if idx == target_idx:
            continue
        if count >= top_k:
            break
        rec_item = all_items_df.iloc[idx]
        recommendations.append({
            'category': rec_item['main_category'],
            'brand': rec_item['brand']
        })
        count += 1
    
    rec_df = pd.DataFrame(recommendations)
    
    unique_categories = rec_df['category'].nunique()
    unique_brands = rec_df['brand'].nunique()
    
    return {
        'query': query,
        'target_category': target_row['main_category'],
        'unique_categories': unique_categories,
        'unique_brands': unique_brands,
        'category_diversity': unique_categories / top_k,
        'brand_diversity': unique_brands / top_k
    }

test_queries = ['laptop', 'headphones', 'camera', 'cable', 'mouse']
diversity_results = []

for query in test_queries:
    result = analyze_diversity(query, top_k=10)
    if result:
        diversity_results.append(result)

diversity_df = pd.DataFrame(diversity_results)

print(f"\n{'='*70}")
print(f"DIVERSITY METRICS")
print(f"{'='*70}")
print(diversity_df.to_string(index=False))

avg_cat_diversity = diversity_df['category_diversity'].mean()
avg_brand_diversity = diversity_df['brand_diversity'].mean()

print(f"\nAverage Category Diversity: {avg_cat_diversity:.2f}")
print(f"Average Brand Diversity: {avg_brand_diversity:.2f}")

# ============================================================================
# STEP 7: VISUALIZATION - EMBEDDING SPACE
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: GENERATING VISUALIZATIONS")
print("=" * 70)

try:
    from sklearn.manifold import TSNE
    
    print("Creating t-SNE visualization...")
    
    # Sample items for visualization
    vis_items = []
    for cat in top_categories:
        cat_items = all_items_df[all_items_df['main_category'] == cat].sample(
            n=min(100, len(all_items_df[all_items_df['main_category'] == cat])),
            random_state=42
        )
        vis_items.append(cat_items)
    
    vis_df = pd.concat(vis_items)
    vis_indices = vis_df.index.tolist()
    vis_embeddings = item_embeddings[vis_indices]
    
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(vis_embeddings)
    
    # Plot
    plt.figure(figsize=(14, 10))
    
    colors = sns.color_palette('husl', len(top_categories))
    
    for idx, cat in enumerate(top_categories):
        mask = vis_df['main_category'] == cat
        plt.scatter(
            tsne_results[mask, 0],
            tsne_results[mask, 1],
            label=cat,
            alpha=0.6,
            s=50,
            color=colors[idx]
        )
    
    plt.title('t-SNE Visualization of Item Embeddings by Category', fontsize=16, fontweight='bold')
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    viz_path = os.path.join(OUTPUT_DIR, 'embedding_visualization.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {viz_path}")
    plt.close()
    
except Exception as e:
    print(f"⚠️  Could not create visualization: {e}")

# ============================================================================
# STEP 8: SUMMARY REPORT
# ============================================================================
# ============================================================================
# STEP 8: SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 70)
print("FINAL EVALUATION REPORT")
print("=" * 70)

# 1. We MUST define the 'report' variable first
report = f"""
RECOMMENDATION SYSTEM EVALUATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET STATISTICS
------------------
- Validation samples: {len(val_df):,}
- Test samples: {len(test_df):,}
- Total items in catalog: {len(all_items_df):,}
- Embedding dimension: {item_embeddings.shape[1]}

RANKING METRICS
---------------
- Hit Rate@{TOP_K}: {hit_rate:.4f} ({hit_rate*100:.2f}%)
- Mean Reciprocal Rank (MRR): {mrr:.4f}
- Samples evaluated: {total_evaluated:,}

CLUSTERING QUALITY
------------------
- Within-category similarity: {within_avg:.4f}
- Between-category similarity: {between_avg:.4f}
- Separation score: {separation:.4f}

DIVERSITY METRICS
-----------------
- Average category diversity: {avg_cat_diversity:.2f}
- Average brand diversity: {avg_brand_diversity:.2f}

OVERALL ASSESSMENT
------------------
"""

# 2. Adding the assessment text
if hit_rate > 0.35 and separation > 0.05:
    report += "EXCELLENT: Your recommendation system is performing very well!\n"
elif hit_rate > 0.25 and separation > 0.03:
    report += "GOOD: Solid performance with room for improvement.\n"
else:
    report += "NEEDS IMPROVEMENT: Consider retraining with more data or better architecture.\n"

# 3. Print it to the console
print(report)

# 4. Save to file using 'utf-8' to avoid the Emoji crash
report_path = os.path.join(OUTPUT_DIR, 'evaluation_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"✓ Report saved to: {report_path}")

# ============================================================================
# COMPLETION
# ============================================================================
print("\n" + "=" * 70)
print("EVALUATION COMPLETE!")
print("=" * 70)
print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")