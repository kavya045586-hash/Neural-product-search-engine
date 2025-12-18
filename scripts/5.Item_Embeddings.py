"""
STEP 3: GENERATE ITEM EMBEDDINGS FOR ALL PRODUCTS
This extracts the Item Tower from your trained model and generates
embeddings for every product in your catalog
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import json
import os
from tensorflow.keras import Model
from datetime import datetime

print("=" * 70)
print("STEP 3: GENERATING ITEM EMBEDDINGS")
print("=" * 70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS TO MATCH YOUR SETUP
# ============================================================================

DATA_DIR = r"C:\Users\nagar\Downloads\parquet\processed_data"
VECTORIZED_DIR = os.path.join(DATA_DIR, "vectorized")
MAPPINGS_DIR = os.path.join(DATA_DIR, "mappings")
MODEL_PATH = os.path.join(VECTORIZED_DIR, "two_tower_model_100k.keras")
OUTPUT_DIR = VECTORIZED_DIR

# ============================================================================
# STEP 1: LOAD THE TRAINED MODEL
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: LOADING TRAINED MODEL")
print("=" * 70)

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at: {MODEL_PATH}")
    print("\nPlease train the model first using Step 2 (Model Building)")
    exit(1)

print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✓ Model loaded: {model.name}")

model.summary()

# ============================================================================
# STEP 2: EXTRACT THE ITEM TOWER
# ============================================================================
# ============================================================================
# STEP 2: EXTRACT THE ITEM TOWER (ROBUST VERSION)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: EXTRACTING ITEM TOWER SUB-MODEL")
print("=" * 70)

# 1. Find the layer we want as output
try:
    item_output = model.get_layer('item_final').output
    print(f"✓ Found output layer: item_final")
except Exception as e:
    print(f"❌ Error finding item_final layer: {e}")
    exit(1)

# 2. Get the specific input tensors by index from the main model
# Based on your previous script, the inputs are in this order:
# [0]: user_input, [1]: item_input, [2]: brand_input, [3]: category_input, [4]: text_input
try:
    item_tower_inputs = [
        model.input[1], # item_input
        model.input[2], # brand_input
        model.input[3], # category_input
        model.input[4]  # text_input
    ]
    print(f"✓ Found item tower inputs")
except Exception as e:
    print(f"❌ Error identifying inputs: {e}")
    print(f"Model actually has {len(model.input)} inputs.")
    exit(1)

# 3. Create the sub-model
try:
    item_tower_model = Model(
        inputs=item_tower_inputs,
        outputs=item_output,
        name="ItemTower"
    )
    print("✓ Item Tower model created successfully!")
except Exception as e:
    print(f"❌ Error creating ItemTower Model: {e}")
    # Fallback: Just print details to debug
    print(f"DEBUG - Inputs: {item_tower_inputs}")
    print(f"DEBUG - Output: {item_output}")
    exit(1)

print(f"  Input shape: {item_tower_model.input_shape}")
print(f"  Output shape: {item_tower_model.output_shape}")

# ============================================================================
# STEP 3: CREATE CATALOG OF ALL UNIQUE ITEMS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: BUILDING ITEM CATALOG")
print("=" * 70)

# Load all data splits to get complete item list
print("Loading all data splits...")
train_df = pd.read_parquet(os.path.join(VECTORIZED_DIR, 'train_final_mini.parquet'))
val_df = pd.read_parquet(os.path.join(VECTORIZED_DIR, 'val_final_mini.parquet'))
test_df = pd.read_parquet(os.path.join(VECTORIZED_DIR, 'test_final_mini.parquet'))

# Combine all data
all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
print(f"✓ Loaded {len(all_data):,} total interactions")

# Get unique items with their features
print("\nExtracting unique items...")
item_columns = ['asin', 'item_id', 'title', 'brand', 'brand_id', 
                'main_category', 'main_category_id', 'title_vector']

# Drop duplicates based on item_id (keeping first occurrence)
all_items_df = all_data[item_columns].drop_duplicates(subset=['item_id']).reset_index(drop=True)

print(f"✓ Found {len(all_items_df):,} unique items")
print(f"\nSample items:")
print(all_items_df[['asin', 'title', 'brand', 'main_category']].head())

# ============================================================================
# STEP 4: GENERATE EMBEDDINGS FOR ALL ITEMS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: GENERATING ITEM EMBEDDINGS")
print("=" * 70)

print(f"Generating embeddings for {len(all_items_df):,} items...")

# Prepare inputs for the Item Tower
item_ids = all_items_df['item_id'].values.astype('int32')
brand_ids = all_items_df['brand_id'].values.astype('int32')
category_ids = all_items_df['main_category_id'].values.astype('int32')
title_vectors = np.stack(all_items_df['title_vector'].values).astype('float32')

print(f"\nInput shapes:")
print(f"  item_ids: {item_ids.shape}")
print(f"  brand_ids: {brand_ids.shape}")
print(f"  category_ids: {category_ids.shape}")
print(f"  title_vectors: {title_vectors.shape}")

# Generate embeddings using the Item Tower
print("\nRunning Item Tower inference...")
item_embeddings = item_tower_model.predict(
    [item_ids, brand_ids, category_ids, title_vectors],
    batch_size=512,
    verbose=1
)

print(f"\n✓ Generated embeddings!")
print(f"  Shape: {item_embeddings.shape}")
print(f"  Expected: ({len(all_items_df)}, 64)")

# Verify embeddings
print(f"\nEmbedding statistics:")
print(f"  Mean: {item_embeddings.mean():.4f}")
print(f"  Std: {item_embeddings.std():.4f}")
print(f"  Min: {item_embeddings.min():.4f}")
print(f"  Max: {item_embeddings.max():.4f}")

# Check for invalid values
nan_count = np.isnan(item_embeddings).sum()
inf_count = np.isinf(item_embeddings).sum()

if nan_count > 0:
    print(f"⚠️  WARNING: {nan_count} NaN values in embeddings")
if inf_count > 0:
    print(f"⚠️  WARNING: {inf_count} Inf values in embeddings")

if nan_count == 0 and inf_count == 0:
    print("✓ No NaN or Inf values detected")

# ============================================================================
# STEP 5: NORMALIZE EMBEDDINGS (OPTIONAL BUT RECOMMENDED)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: NORMALIZING EMBEDDINGS")
print("=" * 70)

print("Normalizing embeddings to unit length...")
# This makes cosine similarity == dot product (faster computation)

norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
item_embeddings_normalized = item_embeddings / (norms + 1e-8)  # Add small epsilon to avoid division by zero

print("✓ Embeddings normalized")
print(f"  New norms - Mean: {np.linalg.norm(item_embeddings_normalized, axis=1).mean():.4f}")
print(f"  (Should be very close to 1.0)")

# Use normalized embeddings for recommendations
item_embeddings = item_embeddings_normalized

# ============================================================================
# STEP 6: SAVE EMBEDDINGS AND ITEM CATALOG
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: SAVING OUTPUTS")
print("=" * 70)

# Save embeddings as numpy array
embeddings_path = os.path.join(OUTPUT_DIR, 'item_embeddings_100k.npy')
np.save(embeddings_path, item_embeddings)
print(f"✓ Saved embeddings to: {embeddings_path}")
print(f"  Size: {os.path.getsize(embeddings_path) / (1024**2):.2f} MB")

# Save item catalog as parquet
catalog_path = os.path.join(OUTPUT_DIR, 'all_items_processed_100k.parquet')
all_items_df.to_parquet(catalog_path, index=False)
print(f"✓ Saved item catalog to: {catalog_path}")
print(f"  Size: {os.path.getsize(catalog_path) / (1024**2):.2f} MB")

# Also save as CSV for easy viewing
csv_path = os.path.join(OUTPUT_DIR, 'all_items_catalog.csv')
all_items_df[['asin', 'title', 'brand', 'main_category']].to_csv(csv_path, index=False)
print(f"✓ Saved readable catalog to: {csv_path}")

# ============================================================================
# STEP 7: VERIFY ALIGNMENT
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: VERIFYING DATA ALIGNMENT")
print("=" * 70)

print("Checking if embeddings align with catalog...")
if len(all_items_df) == len(item_embeddings):
    print(f"✅ PERFECT ALIGNMENT!")
    print(f"   Catalog rows: {len(all_items_df):,}")
    print(f"   Embedding rows: {len(item_embeddings):,}")
else:
    print(f"❌ MISMATCH DETECTED!")
    print(f"   Catalog rows: {len(all_items_df):,}")
    print(f"   Embedding rows: {len(item_embeddings):,}")
    print("   This will cause incorrect recommendations!")
    exit(1)

# Quick sanity check: Generate a test recommendation
print("\n" + "=" * 70)
print("QUICK SANITY CHECK")
print("=" * 70)

print("Testing similarity computation...")

# Pick a random item
test_idx = 0
test_item = all_items_df.iloc[test_idx]

print(f"\nTest item:")
print(f"  Title: {test_item['title']}")
print(f"  Brand: {test_item['brand']}")
print(f"  Category: {test_item['main_category']}")

# Get its embedding
test_embedding = item_embeddings[test_idx].reshape(1, -1)

# Compute similarity with all items
similarities = np.dot(item_embeddings, test_embedding.T).flatten()

# Get top 5 similar items (excluding itself)
similar_indices = np.argsort(similarities)[::-1][1:6]

print(f"\nTop 5 similar items:")
for rank, idx in enumerate(similar_indices, 1):
    similar_item = all_items_df.iloc[idx]
    score = similarities[idx]
    print(f"{rank}. [{score:.4f}] {similar_item['title'][:60]}")
    print(f"   Brand: {similar_item['brand']} | Category: {similar_item['main_category']}")

# ============================================================================
# STEP 8: CREATE INDEX MAPPING FOR FAST LOOKUP
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: CREATING FAST LOOKUP INDEX")
print("=" * 70)

# Create mapping from item_id to index in embeddings array
item_id_to_index = {
    row['item_id']: idx 
    for idx, row in all_items_df.iterrows()
}

# Save this mapping
index_mapping_path = os.path.join(MAPPINGS_DIR, 'item_id_to_index.json')
with open(index_mapping_path, 'w') as f:
    json.dump(item_id_to_index, f)

print(f"✓ Saved index mapping: {index_mapping_path}")
print(f"  Contains {len(item_id_to_index):,} mappings")

# ============================================================================
# COMPLETION
# ============================================================================
print("\n" + "=" * 70)
print("✅ ITEM EMBEDDINGS GENERATION COMPLETE!")
print("=" * 70)
print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n📁 Generated files:")
print(f"  1. {embeddings_path}")
print(f"  2. {catalog_path}")
print(f"  3. {csv_path}")
print(f"  4. {index_mapping_path}")

print("\n📊 Summary:")
print(f"  Total items: {len(all_items_df):,}")
print(f"  Embedding dimension: {item_embeddings.shape[1]}")
print(f"  Total parameters: {len(all_items_df) * item_embeddings.shape[1]:,}")

print("\n📋 NEXT STEPS:")
print("  1. Run Step 4 to build the recommendation app")
print("  2. Test recommendations with different queries")
print("  3. Run evaluation metrics")

print("\n💡 TIP: The embeddings are normalized, so you can use dot product")
print("   for fast similarity search instead of cosine similarity!")