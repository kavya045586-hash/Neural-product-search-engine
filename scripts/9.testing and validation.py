# Test 1: Load everything
import pandas as pd
import numpy as np

all_items_df = pd.read_parquet('processed_data/vectorized/all_items_processed_100k.parquet')
item_embeddings = np.load('processed_data/vectorized/item_embeddings_100k.npy')

print(f"Items: {len(all_items_df)}, Embeddings: {len(item_embeddings)}")
# Should match!

# Test 2: Sample recommendation
query = "laptop"
mask = all_items_df['title'].str.contains(query, case=False, na=False)
target_idx = all_items_df[mask].iloc[0].name

target_vec = item_embeddings[target_idx].reshape(1, -1)
sims = np.dot(item_embeddings, target_vec.T).flatten()
top_5 = np.argsort(sims)[::-1][1:6]

print("\nTop 5 similar items:")
for idx in top_5:
    print(f"- {all_items_df.iloc[idx]['title'][:60]}")