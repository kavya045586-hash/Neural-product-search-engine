import pandas as pd
import numpy as np
import os
from datetime import datetime

print("=" * 70)
print("TEXT VECTORIZATION PIPELINE")
print("=" * 70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = r"C:\Users\nagar\Downloads\parquet\processed_data"
OUTPUT_DIR = os.path.join(DATA_DIR, "vectorized")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# STEP 1: CHECK FOR DEPENDENCIES
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: CHECKING DEPENDENCIES")
print("=" * 70)

try:
    from sentence_transformers import SentenceTransformer
    import torch
    print("✓ sentence-transformers installed")
    print("✓ torch installed")
except ImportError as e:
    print("❌ Missing dependencies!")
    print("\nPlease install required packages:")
    print("  pip install sentence-transformers torch")
    print("\nOr if on CPU only:")
    print("  pip install sentence-transformers")
    exit(1)

# ============================================================================
# STEP 2: LOAD DATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: LOADING PROCESSED DATA")
print("=" * 70)

try:
    train_df = pd.read_parquet(os.path.join(DATA_DIR, 'train_categorical.parquet'))
    val_df = pd.read_parquet(os.path.join(DATA_DIR, 'val_categorical.parquet'))
    test_df = pd.read_parquet(os.path.join(DATA_DIR, 'test_categorical.parquet'))
    
    print(f"✓ Train: {len(train_df):,} rows")
    print(f"✓ Val:   {len(val_df):,} rows")
    print(f"✓ Test:  {len(test_df):,} rows")
except FileNotFoundError:
    print("❌ Processed data files not found!")
    print(f"   Expected location: {DATA_DIR}")
    print("\n   Please run the preprocessing pipeline first.")
    exit(1)

# ============================================================================
# STEP 3: CREATE MINI DATASET (FOR FASTER PROTOTYPING)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: CREATING MINI DATASET")
print("=" * 70)

print("Choose dataset size:")
print("  1. Mini (5% - ~100k rows) - FAST, for testing")
print("  2. Medium (20% - ~400k rows) - BALANCED")
print("  3. Full (100% - 2M rows) - SLOW, best accuracy")

# For automation, let's use 5% by default
# You can change this to 0.20 (20%) or 1.0 (100%) later
SAMPLE_FRACTION = 0.05

print(f"\n⚙️  Using {SAMPLE_FRACTION*100:.0f}% of data")

train_mini = train_df.sample(frac=SAMPLE_FRACTION, random_state=42)
val_mini = val_df.sample(frac=SAMPLE_FRACTION, random_state=42)
test_mini = test_df.sample(frac=SAMPLE_FRACTION, random_state=42)

print(f"✓ Mini train: {len(train_mini):,} rows")
print(f"✓ Mini val:   {len(val_mini):,} rows")
print(f"✓ Mini test:  {len(test_mini):,} rows")

# ============================================================================
# STEP 4: SETUP SENTENCE TRANSFORMER MODEL
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: LOADING SENTENCE TRANSFORMER MODEL")
print("=" * 70)

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

if device == 'cpu':
    print("⚠️  Running on CPU - this will be slower")
    print("   For faster processing, use a GPU-enabled environment")
else:
    print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")

print("\nLoading model 'all-MiniLM-L6-v2'...")
print("  (This creates 384-dimensional vectors)")

model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print("✓ Model loaded successfully!")

# ============================================================================
# STEP 5: VECTORIZE TITLES
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: VECTORIZING PRODUCT TITLES")
print("=" * 70)

def encode_titles(df, column_name='title', batch_size=64):
    """Encode titles to vectors"""
    print(f"  Encoding {len(df):,} titles...")
    
    titles_list = df[column_name].tolist()
    
    # Encode with progress bar
    vectors = model.encode(
        titles_list, 
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_numpy=True
    )
    
    print(f"  ✓ Generated {vectors.shape[0]:,} vectors of dimension {vectors.shape[1]}")
    return list(vectors)

print("\n🔄 Processing TRAIN set...")
train_mini['title_vector'] = encode_titles(train_mini)

print("\n🔄 Processing VAL set...")
val_mini['title_vector'] = encode_titles(val_mini)

print("\n🔄 Processing TEST set...")
test_mini['title_vector'] = encode_titles(test_mini)

print("\n✓ All vectorization complete!")

# ============================================================================
# STEP 6: VERIFY VECTOR QUALITY
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: VERIFYING VECTOR QUALITY")
print("=" * 70)

# Check a sample vector
sample_vector = train_mini['title_vector'].iloc[0]
print(f"Vector shape: {sample_vector.shape}")
print(f"Vector type: {type(sample_vector)}")
print(f"Sample values: {sample_vector[:5]}")

# Check for NaN or Inf
has_nan = any(np.isnan(v).any() for v in train_mini['title_vector'].head(100))
has_inf = any(np.isinf(v).any() for v in train_mini['title_vector'].head(100))

if has_nan or has_inf:
    print("❌ WARNING: Vectors contain NaN or Inf values!")
else:
    print("✓ No NaN or Inf values detected")

# ============================================================================
# STEP 7: SAVE VECTORIZED DATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: SAVING VECTORIZED DATA")
print("=" * 70)

train_mini.to_parquet(
    os.path.join(OUTPUT_DIR, 'train_final_mini.parquet'),
    index=False
)
val_mini.to_parquet(
    os.path.join(OUTPUT_DIR, 'val_final_mini.parquet'),
    index=False
)
test_mini.to_parquet(
    os.path.join(OUTPUT_DIR, 'test_final_mini.parquet'),
    index=False
)

print("✓ Saved vectorized datasets:")
print(f"  {os.path.join(OUTPUT_DIR, 'train_final_mini.parquet')}")
print(f"  {os.path.join(OUTPUT_DIR, 'val_final_mini.parquet')}")
print(f"  {os.path.join(OUTPUT_DIR, 'test_final_mini.parquet')}")

# ============================================================================
# STEP 8: SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

total_vectors = len(train_mini) + len(val_mini) + len(test_mini)
print(f"\nTotal vectors generated: {total_vectors:,}")
print(f"Vector dimension: 384")
print(f"Model used: all-MiniLM-L6-v2")

print("\nDataset sizes:")
print(f"  Train: {len(train_mini):,} ({len(train_mini)/total_vectors:.1%})")
print(f"  Val:   {len(val_mini):,} ({len(val_mini)/total_vectors:.1%})")
print(f"  Test:  {len(test_mini):,} ({len(test_mini)/total_vectors:.1%})")

print("\nMemory estimates:")
train_size_mb = (len(train_mini) * 384 * 4) / (1024**2)  # 4 bytes per float32
print(f"  Train vectors: ~{train_size_mb:.1f} MB")

# ============================================================================
# COMPLETION
# ============================================================================
print("\n" + "=" * 70)
print("✅ TEXT VECTORIZATION COMPLETE!")
print("=" * 70)
print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n📋 NEXT STEPS:")
print("  1. Train the Two-Tower model")
print("  2. Generate item embeddings")
print("  3. Build recommendation engine")

print("\n💡 TIP: To use more data, change SAMPLE_FRACTION to 0.20 or 1.0")
print("   Larger datasets = better accuracy but slower training")