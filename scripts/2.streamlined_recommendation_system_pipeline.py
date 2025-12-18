import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================

INPUT_FILE = r"C:\Users\nagar\Downloads\parquet\electronics_sample_2M (1).parquet"
OUTPUT_DIR = r"C:\Users\nagar\Downloads\parquet\processed_data"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "mappings"), exist_ok=True)

print("=" * 70)
print("RECOMMENDATION SYSTEM - COMPLETE PIPELINE")
print("=" * 70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: LOADING DATA")
print("=" * 70)

df = pd.read_parquet(INPUT_FILE)
print(f"✓ Loaded {len(df):,} reviews")
print(f"  Columns: {df.columns.tolist()}")

# ============================================================================
# STEP 2: CREATE TARGET VARIABLE
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: CREATING TARGET VARIABLE")
print("=" * 70)

# Create binary target: 1 if rating >= 4.0, else 0
df['positive'] = (df['overall'] >= 4.0).astype(int)

print("Target distribution:")
print(df['positive'].value_counts(normalize=True))
print(f"Positive rate: {df['positive'].mean():.2%}")

# ============================================================================
# STEP 3: HANDLE MISSING VALUES
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: HANDLING MISSING VALUES")
print("=" * 70)

print("\nBefore cleaning:")
print(df.isnull().sum())

# Fill missing titles with "Unknown Product"
if 'title' in df.columns:
    df['title'] = df['title'].fillna("Unknown Product")

# Fill missing brands with "Unknown Brand"
if 'brand' in df.columns:
    df['brand'] = df['brand'].fillna("Unknown Brand")

print("\nAfter cleaning:")
print(df.isnull().sum())

# ============================================================================
# STEP 4: EXTRACT MAIN CATEGORY
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: EXTRACTING MAIN CATEGORY")
print("=" * 70)

import numpy as np
import pandas as pd
import ast

def get_most_specific_category(val):
    """Extract the most specific category from nested list/array/string"""
    
    # Handle None/NaN (only scalars)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 'Unknown'
    
    # Handle numpy arrays
    if isinstance(val, np.ndarray):
        val = val.tolist()
    
    # Handle strings that look like lists
    if isinstance(val, str) and val.startswith('['):
        try:
            val = ast.literal_eval(val)
        except Exception:
            return 'Unknown'
    
    # Must be a list now
    if not isinstance(val, list):
        return 'Unknown'
    
    if len(val) == 0:
        return 'Unknown'
    
    try:
        # Check for nested list [[...]]
        first_item = val[0]
        if isinstance(first_item, (list, np.ndarray)):
            # Nested: get last item of last sublist
            last_sublist = val[-1]
            if isinstance(last_sublist, (list, np.ndarray)) and len(last_sublist) > 0:
                return last_sublist[-1]
            return 'Unknown'
        else:
            # Flat list: get last item
            return val[-1]
    except Exception:
        return 'Unknown'

# Apply safely
if 'categories' in df.columns:
    df['main_category'] = df['categories'].apply(get_most_specific_category)
    print(f"✓ Extracted {df['main_category'].nunique()} unique categories")
    print("\nTop 10 categories:")
    print(df['main_category'].value_counts().head(10))
else:
    print("⚠️  No 'categories' column found, using 'Unknown' for all")
    df['main_category'] = 'Unknown'

# ============================================================================
# STEP 5: TIME-BASED TRAIN/VAL/TEST SPLIT
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: CREATING TRAIN/VAL/TEST SPLITS")
print("=" * 70)

# Sort by time
df_sorted = df.sort_values(by='unixReviewTime').reset_index(drop=True)

# Calculate split indices
total_rows = len(df_sorted)
train_end = int(total_rows * 0.8)
val_end = int(total_rows * 0.9)

# Create splits
train_df = df_sorted.iloc[:train_end].copy()
val_df = df_sorted.iloc[train_end:val_end].copy()
test_df = df_sorted.iloc[val_end:].copy()

print(f"Train set: {len(train_df):,} rows ({len(train_df)/total_rows:.1%})")
print(f"Val set:   {len(val_df):,} rows ({len(val_df)/total_rows:.1%})")
print(f"Test set:  {len(test_df):,} rows ({len(test_df)/total_rows:.1%})")

# ============================================================================
# STEP 6: CREATE CATEGORICAL MAPPINGS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: CREATING CATEGORICAL MAPPINGS")
print("=" * 70)

def create_mapping(dataframe, column_name, output_filename):
    """Create ID mapping for a categorical column"""
    print(f"  Creating mapping for: {column_name}")
    
    # Get unique values (drop NaN)
    unique_values = dataframe[column_name].dropna().unique()
    
    # Create mapping (starting at ID 2)
    mapping = {val: i+2 for i, val in enumerate(unique_values)}
    
    # Add special tokens
    mapping['<UNKNOWN>'] = 0  # For values not in training
    mapping['<MISSING>'] = 1  # For NaN values
    
    # Save to JSON
    with open(output_filename, 'w') as f:
        json.dump(mapping, f, default=lambda o: int(o) if isinstance(o, np.integer) else o)
    
    print(f"    Saved {len(mapping):,} entries to {output_filename}")
    return mapping

# Create mappings from TRAINING DATA ONLY
mappings_dir = os.path.join(OUTPUT_DIR, "mappings")

user_mapping = create_mapping(train_df, 'reviewerID', 
                              os.path.join(mappings_dir, 'user_mapping.json'))
item_mapping = create_mapping(train_df, 'asin', 
                              os.path.join(mappings_dir, 'item_mapping.json'))
brand_mapping = create_mapping(train_df, 'brand', 
                               os.path.join(mappings_dir, 'brand_mapping.json'))
category_mapping = create_mapping(train_df, 'main_category', 
                                  os.path.join(mappings_dir, 'category_mapping.json'))

print(f"\n✓ Created all mappings")
print(f"  Users: {len(user_mapping):,}")
print(f"  Items: {len(item_mapping):,}")
print(f"  Brands: {len(brand_mapping):,}")
print(f"  Categories: {len(category_mapping):,}")

# ============================================================================
# STEP 7: APPLY MAPPINGS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: APPLYING MAPPINGS TO DATA")
print("=" * 70)

def apply_mapping(dataframe, column_name, mapping):
    """Apply mapping with proper handling of unknown/missing values"""
    def get_id(val):
        if pd.isna(val) or val == "":
            return mapping['<MISSING>']
        if val in mapping:
            return mapping[val]
        return mapping['<UNKNOWN>']
    
    return dataframe[column_name].apply(get_id)

# Apply to all datasets
for dataset_name, dataset in [('train', train_df), ('val', val_df), ('test', test_df)]:
    print(f"  Processing {dataset_name} set...")
    dataset['user_id'] = apply_mapping(dataset, 'reviewerID', user_mapping)
    dataset['item_id'] = apply_mapping(dataset, 'asin', item_mapping)
    dataset['brand_id'] = apply_mapping(dataset, 'brand', brand_mapping)
    dataset['main_category_id'] = apply_mapping(dataset, 'main_category', category_mapping)

print("✓ Mappings applied to all datasets")

# ============================================================================
# STEP 8: SAVE PROCESSED DATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: SAVING PROCESSED DATA")
print("=" * 70)

# Select final columns
final_columns = ['reviewerID', 'asin', 'overall', 'unixReviewTime', 
                'title', 'brand', 'main_category', 'categories',
                'user_id', 'item_id', 'brand_id', 'main_category_id', 'positive']

train_df[final_columns].to_parquet(
    os.path.join(OUTPUT_DIR, 'train_categorical.parquet'), 
    index=False
)
val_df[final_columns].to_parquet(
    os.path.join(OUTPUT_DIR, 'val_categorical.parquet'), 
    index=False
)
test_df[final_columns].to_parquet(
    os.path.join(OUTPUT_DIR, 'test_categorical.parquet'), 
    index=False
)

print("✓ Saved all processed files:")
print(f"  {os.path.join(OUTPUT_DIR, 'train_categorical.parquet')}")
print(f"  {os.path.join(OUTPUT_DIR, 'val_categorical.parquet')}")
print(f"  {os.path.join(OUTPUT_DIR, 'test_categorical.parquet')}")

# ============================================================================
# STEP 9: SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 9: SUMMARY STATISTICS")
print("=" * 70)

print("\nDataset Statistics:")
print(f"  Total reviews: {len(df):,}")
print(f"  Unique users: {df['reviewerID'].nunique():,}")
print(f"  Unique items: {df['asin'].nunique():,}")
print(f"  Unique brands: {df['brand'].nunique():,}")
print(f"  Unique categories: {df['main_category'].nunique():,}")

print("\nPositive Rate by Split:")
print(f"  Train: {train_df['positive'].mean():.2%}")
print(f"  Val:   {val_df['positive'].mean():.2%}")
print(f"  Test:  {test_df['positive'].mean():.2%}")

print("\nCold Start Statistics:")
train_users = set(train_df['reviewerID'].unique())
train_items = set(train_df['asin'].unique())

val_cold_users = sum(1 for u in val_df['reviewerID'].unique() if u not in train_users)
val_cold_items = sum(1 for i in val_df['asin'].unique() if i not in train_items)

print(f"  Val cold start users: {val_cold_users:,} ({val_cold_users/val_df['reviewerID'].nunique():.1%})")
print(f"  Val cold start items: {val_cold_items:,} ({val_cold_items/val_df['asin'].nunique():.1%})")

# ============================================================================
# COMPLETION
# ============================================================================
print("\n" + "=" * 70)
print("✅ PIPELINE COMPLETE!")
print("=" * 70)
print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n📁 Output files saved to:")
print(f"  {OUTPUT_DIR}")

print("\n📋 NEXT STEPS:")
print("  1. Run text vectorization script")
print("  2. Train the Two-Tower model")
print("  3. Generate item embeddings")
print("  4. Test recommendations")

print("\n💡 TIP: Keep the 'mappings' folder - you'll need it for inference!")