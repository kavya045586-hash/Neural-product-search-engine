"""
TRAIN TWO-TOWER RECOMMENDATION MODEL
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import json
import os
from tensorflow.keras import layers, Model, Input, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from datetime import datetime

print("=" * 70)
print("TWO-TOWER MODEL TRAINING")
print("=" * 70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"TensorFlow Version: {tf.__version__}")


# --- ADD THIS AFTER IMPORTS ---
MODEL_SAVE_PATH = r"C:\Users\nagar\Downloads\parquet\processed_data\vectorized\two_tower_model_100k.keras"

# Check if we already have a trained model
if os.path.exists(MODEL_SAVE_PATH):
    print("✓ Found existing model. Loading...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    skip_training = True
else:
    skip_training = False

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = r"C:\Users\nagar\Downloads\parquet\processed_data"
VECTORIZED_DIR = os.path.join(DATA_DIR, "vectorized")
MAPPINGS_DIR = os.path.join(DATA_DIR, "mappings")

# Create directories
os.makedirs(VECTORIZED_DIR, exist_ok=True)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: LOADING DATA")
print("=" * 70)

try:
    print("Loading train data...")
    train_df = pd.read_parquet(os.path.join(VECTORIZED_DIR, 'train_final_mini.parquet'))
    print(f"✓ Train: {len(train_df):,} rows")
    
    print("Loading validation data...")
    val_df = pd.read_parquet(os.path.join(VECTORIZED_DIR, 'val_final_mini.parquet'))
    print(f"✓ Val: {len(val_df):,} rows")
    
    print("Loading test data...")
    test_df = pd.read_parquet(os.path.join(VECTORIZED_DIR, 'test_final_mini.parquet'))
    print(f"✓ Test: {len(test_df):,} rows")
    
except FileNotFoundError as e:
    print(f"\n❌ ERROR: Could not find vectorized data files!")
    print(f"   {e}")
    print("\n   Please run text_vectorization.py first!")
    exit(1)

# ============================================================================
# STEP 2: LOAD VOCABULARIES
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: LOADING VOCABULARIES")
print("=" * 70)

def load_mapping(filename):
    with open(filename, 'r') as f:
        return json.load(f)

try:
    user_mapping = load_mapping(os.path.join(MAPPINGS_DIR, 'user_mapping.json'))
    item_mapping = load_mapping(os.path.join(MAPPINGS_DIR, 'item_mapping.json'))
    brand_mapping = load_mapping(os.path.join(MAPPINGS_DIR, 'brand_mapping.json'))
    category_mapping = load_mapping(os.path.join(MAPPINGS_DIR, 'category_mapping.json'))
    
    num_users = len(user_mapping) + 1
    num_items = len(item_mapping) + 1
    num_brands = len(brand_mapping) + 1
    num_categories = len(category_mapping) + 1
    
    print(f"✓ Users: {num_users:,}")
    print(f"✓ Items: {num_items:,}")
    print(f"✓ Brands: {num_brands:,}")
    print(f"✓ Categories: {num_categories:,}")
    
except FileNotFoundError as e:
    print(f"\n❌ ERROR: Could not find mapping files!")
    print(f"   {e}")
    exit(1)

# ============================================================================
# STEP 3: PREPARE DATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: PREPARING DATA")
print("=" * 70)

def prepare_data(df):
    """Convert DataFrame to model inputs"""
    user_ids = df['user_id'].values.astype('int32')
    item_ids = df['item_id'].values.astype('int32')
    brand_ids = df['brand_id'].values.astype('int32')
    category_ids = df['main_category_id'].values.astype('int32')
    title_vectors = np.stack(df['title_vector'].values).astype('float32')
    
    X = {
        "user_input": user_ids,
        "item_input": item_ids,
        "brand_input": brand_ids,
        "category_input": category_ids,
        "text_input": title_vectors
    }
    
    y = df['positive'].values.astype('float32')
    return X, y

print("Preparing train data...")
X_train, y_train = prepare_data(train_df)

print("Preparing validation data...")
X_val, y_val = prepare_data(val_df)

print("Preparing test data...")
X_test, y_test = prepare_data(test_df)

print(f"\n✓ Train samples: {len(y_train):,}")
print(f"✓ Val samples: {len(y_val):,}")
print(f"✓ Test samples: {len(y_test):,}")
print(f"✓ Positive rate: {y_train.mean():.2%}")

# ============================================================================
# STEP 4: BUILD MODEL
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: BUILDING TWO-TOWER MODEL")
print("=" * 70)

def build_two_tower_model(
    num_users, num_items, num_brands, num_categories,
    embedding_dim=64,
    vector_dim=384,
    dropout_rate=0.3,
    l2_reg=1e-6
):
    """
    Two-Tower Recommendation Model
    - User Tower: Learns user preferences
    - Item Tower: Learns item characteristics
    """
    
    # ========== USER TOWER ==========
    user_input = Input(shape=(1,), name='user_input')
    user_embedding = layers.Embedding(
        num_users, embedding_dim,
        embeddings_regularizer=regularizers.l2(l2_reg),
        name='user_emb'
    )(user_input)
    user_vec = layers.Flatten(name='user_flatten')(user_embedding)
    
    # Dense layers for user
    user_vec = layers.Dense(128, activation='relu',
                            kernel_regularizer=regularizers.l2(l2_reg))(user_vec)
    user_vec = layers.BatchNormalization()(user_vec)
    user_vec = layers.Dropout(dropout_rate)(user_vec)
    user_vec = layers.Dense(embedding_dim, activation='relu', 
                            name='user_final',
                            kernel_regularizer=regularizers.l2(l2_reg))(user_vec)
    
    # ========== ITEM TOWER ==========
    # Item ID embedding
    item_input = Input(shape=(1,), name='item_input')
    item_emb = layers.Embedding(
        num_items, embedding_dim,
        embeddings_regularizer=regularizers.l2(l2_reg),
        name='item_emb'
    )(item_input)
    item_vec = layers.Flatten()(item_emb)
    
    # Brand embedding
    brand_input = Input(shape=(1,), name='brand_input')
    brand_emb = layers.Embedding(
        num_brands, 32,
        embeddings_regularizer=regularizers.l2(l2_reg),
        name='brand_emb'
    )(brand_input)
    brand_vec = layers.Flatten()(brand_emb)
    
    # Category embedding
    category_input = Input(shape=(1,), name='category_input')
    category_emb = layers.Embedding(
        num_categories, 32,
        embeddings_regularizer=regularizers.l2(l2_reg),
        name='category_emb'
    )(category_input)
    category_vec = layers.Flatten()(category_emb)
    
    # Text vector (from SentenceTransformers)
    text_input = Input(shape=(vector_dim,), name='text_input')
    text_vec = layers.Dense(128, activation='relu',
                           kernel_regularizer=regularizers.l2(l2_reg))(text_input)
    text_vec = layers.BatchNormalization()(text_vec)
    text_vec = layers.Dropout(dropout_rate)(text_vec)
    text_vec = layers.Dense(64, activation='relu', 
                           name='text_dense')(text_vec)
    
    # Combine all item features
    item_merged = layers.Concatenate(name='item_concat')([
        item_vec, brand_vec, category_vec, text_vec
    ])
    
    # Deep fusion of item features
    item_fused = layers.Dense(256, activation='relu',
                             kernel_regularizer=regularizers.l2(l2_reg))(item_merged)
    item_fused = layers.BatchNormalization()(item_fused)
    item_fused = layers.Dropout(dropout_rate)(item_fused)
    
    item_fused = layers.Dense(128, activation='relu',
                             kernel_regularizer=regularizers.l2(l2_reg))(item_fused)
    item_fused = layers.BatchNormalization()(item_fused)
    item_fused = layers.Dropout(dropout_rate)(item_fused)
    
    # Final item vector (must match user vector dimension)
    item_final_vec = layers.Dense(
        embedding_dim, activation='relu',
        name='item_final',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(item_fused)
    
    # ========== MATCHING ==========
    # Compute similarity via dot product
    dot_product = layers.Dot(axes=1, name='similarity')([user_vec, item_final_vec])
    
    # Final prediction
    output = layers.Dense(1, activation='sigmoid', name='output')(dot_product)
    
    # Build model
    model = Model(
        inputs=[user_input, item_input, brand_input, category_input, text_input],
        outputs=output,
        name="TwoTowerRecommender"
    )
    
    return model

print("Building model...")
model = build_two_tower_model(
    num_users=num_users,
    num_items=num_items,
    num_brands=num_brands,
    num_categories=num_categories,
    embedding_dim=64,
    dropout_rate=0.3
)

print("✓ Model architecture created!")

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

print("\n✓ Model compiled!")
print("\nModel Summary:")
model.summary()

# ============================================================================
# STEP 5: SETUP CALLBACKS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: SETTING UP TRAINING CALLBACKS")
print("=" * 70)

MODEL_SAVE_PATH = os.path.join(VECTORIZED_DIR, 'two_tower_model_100k.keras')
BEST_MODEL_PATH = os.path.join(VECTORIZED_DIR, 'best_model.keras')

callbacks = [
    EarlyStopping(
        monitor='val_auc',
        patience=5,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

print("✓ Callbacks configured:")
print("  • Early stopping (patience=5)")
print("  • Learning rate reduction (patience=3)")
print("  • Model checkpoint (best validation AUC)")

# ============================================================================
# STEP 6: TRAIN MODEL
# ============================================================================
# print("\n" + "=" * 70)
# print("STEP 6: TRAINING MODEL")
# print("=" * 70)
# print("\n⚙️  Training Configuration:")
# print(f"  • Max epochs: 30")
# print(f"  • Batch size: 512")
# print(f"  • Initial learning rate: 0.001")
# print(f"  • Optimizer: Adam")

# print("\n🚀 Starting training...\n")

# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=30,
#     batch_size=512,
#     callbacks=callbacks,
#     verbose=1
# )

# print("\n✅ Training complete!")

# ============================================================================
# STEP 7: EVALUATE ON TEST SET
# ============================================================================
# ============================================================================
# STEP 7: EVALUATE ON TEST SET (FIXED)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: EVALUATING ON TEST SET")
print("=" * 70)

# return_dict=True returns a dictionary: {'loss': 0.5, 'auc': 0.8, ...}
test_results = model.evaluate(X_test, y_test, verbose=1, return_dict=True)

print("\n" + "=" * 70)
print("📊 FINAL TEST RESULTS")
print("=" * 70)
for metric_name, value in test_results.items():
    print(f"  {metric_name.upper()}: {value:.4f}")

# Safely extract AUC even if it's named 'auc' or 'auc_1'
test_auc = test_results.get('auc') or test_results.get('auc_1')

if test_auc is None:
    print("\n❌ Error: Could not find AUC in results. Available keys:", list(test_results.keys()))
elif test_auc > 0.85:
    print("\n✅ EXCELLENT MODEL! Ready for production!")
elif test_auc > 0.80:
    print("\n✓ GOOD MODEL! Solid performance!")
elif test_auc > 0.75:
    print("\n⚠️  FAIR MODEL. Consider training longer or with more data.")
else:
    print("\n❌ MODEL NEEDS IMPROVEMENT. Check data quality.")

# ============================================================================
# STEP 8: SAVE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: SAVING MODEL")
print("=" * 70)

print(f"Saving model to: {MODEL_SAVE_PATH}")
model.save(MODEL_SAVE_PATH)
print("✓ Model saved!")

file_size_mb = os.path.getsize(MODEL_SAVE_PATH) / (1024**2)
print(f"  File size: {file_size_mb:.2f} MB")

# ============================================================================
# # ============================================================================
# STEP 9: GENERATING TRAINING PLOTS (FIXED)
# ============================================================================
# We check if 'history' exists. If it does, we plot. If not, we skip.
if 'history' in locals():
    print("\n" + "=" * 70)
    print("STEP 9: GENERATING TRAINING PLOTS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss', color='#3498db')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', color='#e74c3c')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].legend()

    # AUC
    axes[0, 1].plot(history.history['auc'], label='Train AUC', color='#3498db')
    axes[0, 1].plot(history.history['val_auc'], label='Val AUC', color='#e74c3c')
    axes[0, 1].set_title('Model AUC')
    axes[0, 1].legend()

    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train Precision', color='#3498db')
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision', color='#e74c3c')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].legend()

    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train Recall', color='#3498db')
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall', color='#e74c3c')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].legend()

    plt.tight_layout()
    plot_path = os.path.join(VECTORIZED_DIR, 'training_history.png')
    plt.savefig(plot_path, dpi=300)
    print(f"✓ Saved plot: {plot_path}")
    plt.show() # This will pop the window up for you to see
else:
    print("\n⚠️ Skipping Step 9: No training history found in this session.")
    print("💡 You can find your previous graph here: " + os.path.join(VECTORIZED_DIR, 'training_history.png'))
# ============================================================================
# STEP 10: FINAL SUMMARY
# ============================================================================
# ============================================================================
# STEP 10: FINAL SUMMARY (FIXED)
# ============================================================================
print("\n" + "=" * 70)
print("✅ SCRIPT COMPLETE!")
print("=" * 70)
print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n📁 Output Files:")
print(f"  1. {MODEL_SAVE_PATH}")
print(f"  2. {BEST_MODEL_PATH}")

# Only try to print plot_path if it was actually created in Step 9
if 'plot_path' in locals():
    print(f"  3. {plot_path}")
else:
    # If Step 9 was skipped, we just show the expected location
    expected_plot = os.path.join(VECTORIZED_DIR, 'training_history.png')
    print(f"  3. {expected_plot} (Existing plot)")

# Only try to print AUC if history exists
if 'history' in locals():
    print("\n📊 Best Validation AUC: {:.4f}".format(max(history.history['val_auc'])))
else:
    print("\n📊 Evaluation complete (Model loaded from disk).")