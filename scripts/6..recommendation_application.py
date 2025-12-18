"""
STEP 4: RECOMMENDATION APPLICATION
Interactive recommendation system with multiple search methods
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

print("=" * 70)
print("RECOMMENDATION SYSTEM - READY TO USE!")
print("=" * 70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = r"C:\Users\nagar\Downloads\parquet\processed_data"
VECTORIZED_DIR = os.path.join(DATA_DIR, "vectorized")

# ============================================================================
# STEP 1: LOAD ALL NECESSARY DATA
# ============================================================================
print("\n" + "=" * 70)
print("LOADING RECOMMENDATION ENGINE DATA")
print("=" * 70)

# Load item catalog
catalog_path = os.path.join(VECTORIZED_DIR, 'all_items_processed_100k.parquet')
print(f"Loading item catalog from: {catalog_path}")

if not os.path.exists(catalog_path):
    print(f"❌ Catalog not found! Please run Step 3 first.")
    exit(1)

all_items_df = pd.read_parquet(catalog_path)
print(f"✓ Loaded {len(all_items_df):,} items")

# Load embeddings
embeddings_path = os.path.join(VECTORIZED_DIR, 'item_embeddings_100k.npy')
print(f"Loading embeddings from: {embeddings_path}")

if not os.path.exists(embeddings_path):
    print(f"❌ Embeddings not found! Please run Step 3 first.")
    exit(1)

item_embeddings = np.load(embeddings_path)
print(f"✓ Loaded embeddings: {item_embeddings.shape}")

# Verify alignment
if len(all_items_df) != len(item_embeddings):
    print(f"❌ CRITICAL ERROR: Data misalignment!")
    print(f"   Catalog: {len(all_items_df)}, Embeddings: {len(item_embeddings)}")
    exit(1)

print("✓ Data alignment verified")

# ============================================================================
# STEP 2: BUILD RECOMMENDATION FUNCTIONS
# ============================================================================
print("\n" + "=" * 70)
print("BUILDING RECOMMENDATION FUNCTIONS")
print("=" * 70)

def search_items(query, max_results=10):
    """
    Search for items by title/keyword
    
    Args:
        query (str): Search query
        max_results (int): Maximum number of results
    
    Returns:
        DataFrame: Matching items
    """
    mask = all_items_df['title'].str.contains(query, case=False, na=False)
    results = all_items_df[mask].head(max_results)
    return results


def recommend_similar_items(query, top_k=10, method='text'):
    """
    Find items similar to the query
    
    Args:
        query (str): Product name/keyword to search for
        top_k (int): Number of recommendations
        method (str): 'text' for title search, 'category' for category-based
    
    Returns:
        DataFrame: Top K recommendations with scores
    """
    
    # Find matching items
    mask = all_items_df['title'].str.contains(query, case=False, na=False)
    matches = all_items_df[mask]
    
    if len(matches) == 0:
        print(f"❌ No items found matching '{query}'")
        return None
    
    # Get the first match (best match)
    target_row = matches.iloc[0]
    target_idx = target_row.name
    
    print(f"\n{'='*70}")
    print(f"QUERY: '{query}'")
    print(f"{'='*70}")
    print(f"\n📦 Selected Item:")
    print(f"  Title: {target_row['title']}")
    print(f"  Brand: {target_row['brand']}")
    print(f"  Category: {target_row['main_category']}")
    print(f"  ASIN: {target_row['asin']}")
    
    # Get target item's embedding
    target_vector = item_embeddings[target_idx].reshape(1, -1)
    
    # Calculate similarity scores with ALL items
    sim_scores = np.dot(item_embeddings, target_vector.T).flatten()
    
    # Sort by similarity (descending)
    sorted_indices = np.argsort(sim_scores)[::-1]
    
    # Collect top K recommendations (excluding the query item itself)
    recommendations = []
    count = 0
    
    for idx in sorted_indices:
        if idx == target_idx:
            continue  # Skip the query item itself
        
        if count >= top_k:
            break
        
        rec_item = all_items_df.iloc[idx]
        score = sim_scores[idx]
        
        recommendations.append({
            'rank': count + 1,
            'score': score,
            'asin': rec_item['asin'],
            'title': rec_item['title'],
            'brand': rec_item['brand'],
            'category': rec_item['main_category'],
            'item_id': rec_item['item_id']
        })
        
        count += 1
    
    # Convert to DataFrame
    rec_df = pd.DataFrame(recommendations)
    
    # Print recommendations
    print(f"\n{'='*70}")
    print(f"✨ TOP {top_k} RECOMMENDATIONS")
    print(f"{'='*70}\n")
    
    for idx, row in rec_df.iterrows():
        print(f"{row['rank']}. 🎯 Score: {row['score']:.4f}")
        print(f"   📦 {row['title'][:80]}")
        print(f"   🏷️  Brand: {row['brand']}")
        print(f"   📁 Category: {row['category']}")
        print()
    
    # Analyze results
    print(f"\n{'='*70}")
    print(f"📊 RECOMMENDATION ANALYSIS")
    print(f"{'='*70}")
    
    target_category = target_row['main_category']
    same_category = rec_df[rec_df['category'] == target_category]
    
    print(f"Target Category: {target_category}")
    print(f"Same Category: {len(same_category)}/{top_k} ({len(same_category)/top_k*100:.1f}%)")
    print(f"Average Score: {rec_df['score'].mean():.4f}")
    print(f"Score Range: {rec_df['score'].min():.4f} - {rec_df['score'].max():.4f}")
    
    # Category diversity
    unique_categories = rec_df['category'].nunique()
    print(f"Category Diversity: {unique_categories} unique categories")
    
    if len(same_category) / top_k >= 0.7:
        print("✅ Good category consistency!")
    else:
        print("⚠️  Low category consistency - model may need improvement")
    
    return rec_df


def recommend_by_category(category, top_k=10):
    """
    Get top items from a specific category
    
    Args:
        category (str): Category name
        top_k (int): Number of items to return
    
    Returns:
        DataFrame: Top items from category
    """
    category_items = all_items_df[
        all_items_df['main_category'].str.contains(category, case=False, na=False)
    ].head(top_k)
    
    if len(category_items) == 0:
        print(f"❌ No items found in category '{category}'")
        return None
    
    print(f"\n{'='*70}")
    print(f"TOP {len(category_items)} ITEMS IN CATEGORY: {category}")
    print(f"{'='*70}\n")
    
    for idx, row in category_items.iterrows():
        print(f"{idx+1}. {row['title'][:80]}")
        print(f"   Brand: {row['brand']}")
        print()
    
    return category_items


def get_popular_items(top_k=20):
    """
    Get most popular items (based on appearance in data)
    
    Args:
        top_k (int): Number of items
    
    Returns:
        DataFrame: Popular items
    """
    # Sample from different categories for diversity
    categories = all_items_df['main_category'].value_counts().head(5).index
    
    popular = []
    for cat in categories:
        cat_items = all_items_df[all_items_df['main_category'] == cat].head(4)
        popular.append(cat_items)
    
    popular_df = pd.concat(popular).head(top_k)
    
    print(f"\n{'='*70}")
    print(f"🔥 TOP {len(popular_df)} POPULAR ITEMS")
    print(f"{'='*70}\n")
    
    for idx, row in popular_df.iterrows():
        print(f"• {row['title'][:70]}")
        print(f"  Category: {row['main_category']} | Brand: {row['brand']}")
        print()
    
    return popular_df


def compare_items(query1, query2):
    """
    Compare two items and show their similarity
    
    Args:
        query1 (str): First item query
        query2 (str): Second item query
    """
    # Find both items
    mask1 = all_items_df['title'].str.contains(query1, case=False, na=False)
    mask2 = all_items_df['title'].str.contains(query2, case=False, na=False)
    
    matches1 = all_items_df[mask1]
    matches2 = all_items_df[mask2]
    
    if len(matches1) == 0:
        print(f"❌ No items found for '{query1}'")
        return
    
    if len(matches2) == 0:
        print(f"❌ No items found for '{query2}'")
        return
    
    item1 = matches1.iloc[0]
    item2 = matches2.iloc[0]
    
    idx1 = item1.name
    idx2 = item2.name
    
    # Calculate similarity
    vec1 = item_embeddings[idx1]
    vec2 = item_embeddings[idx2]
    similarity = np.dot(vec1, vec2)
    
    print(f"\n{'='*70}")
    print(f"ITEM COMPARISON")
    print(f"{'='*70}\n")
    
    print(f"Item 1: {item1['title']}")
    print(f"  Brand: {item1['brand']} | Category: {item1['main_category']}\n")
    
    print(f"Item 2: {item2['title']}")
    print(f"  Brand: {item2['brand']} | Category: {item2['main_category']}\n")
    
    print(f"Similarity Score: {similarity:.4f}")
    
    if similarity > 0.8:
        print("✅ Very similar items!")
    elif similarity > 0.6:
        print("⚠️  Somewhat similar")
    else:
        print("❌ Not very similar")

print("✓ Recommendation functions ready!")

# ============================================================================
# STEP 3: DISPLAY SYSTEM INFORMATION
# ============================================================================
print("\n" + "=" * 70)
print("SYSTEM INFORMATION")
print("=" * 70)

print(f"\n📊 Catalog Statistics:")
print(f"  Total Items: {len(all_items_df):,}")
print(f"  Unique Brands: {all_items_df['brand'].nunique():,}")
print(f"  Unique Categories: {all_items_df['main_category'].nunique():,}")
print(f"  Embedding Dimension: {item_embeddings.shape[1]}")

print(f"\n📁 Top 10 Categories:")
top_categories = all_items_df['main_category'].value_counts().head(10)
for cat, count in top_categories.items():
    print(f"  • {cat}: {count:,} items")

print(f"\n🏷️  Top 10 Brands:")
top_brands = all_items_df['brand'].value_counts().head(10)
for brand, count in top_brands.items():
    print(f"  • {brand}: {count:,} items")

# ============================================================================
# STEP 4: RUN TEST QUERIES
# ============================================================================
print("\n" + "=" * 70)
print("RUNNING TEST QUERIES")
print("=" * 70)

# Test Query 1: Laptops
print("\n" + "🔍 TEST 1: LAPTOP RECOMMENDATIONS")
print("-" * 70)
rec1 = recommend_similar_items("laptop", top_k=5)

# Test Query 2: Headphones
print("\n" + "🔍 TEST 2: HEADPHONE RECOMMENDATIONS")
print("-" * 70)
rec2 = recommend_similar_items("headphones", top_k=5)

# Test Query 3: Camera
print("\n" + "🔍 TEST 3: CAMERA RECOMMENDATIONS")
print("-" * 70)
rec3 = recommend_similar_items("camera", top_k=5)

# Test Query 4: Cable
print("\n" + "🔍 TEST 4: CABLE RECOMMENDATIONS")
print("-" * 70)
rec4 = recommend_similar_items("cable", top_k=5)

# ============================================================================
# STEP 5: INTERACTIVE MODE
# ============================================================================
print("\n" + "=" * 70)
print("🎮 INTERACTIVE RECOMMENDATION MODE")
print("=" * 70)

print("\nAvailable commands:")
print("  1. recommend('product name') - Get recommendations")
print("  2. search('keyword') - Search products")
print("  3. category('category name') - Browse category")
print("  4. compare('item1', 'item2') - Compare two items")
print("  5. popular() - Show popular items")
print("  6. exit - Exit interactive mode")

def interactive_mode():
    """Interactive recommendation loop"""
    print("\nStarting interactive mode...")
    print("Type your command or 'exit' to quit\n")
    
    while True:
        try:
            user_input = input("🔍 > ").strip()
            
            if user_input.lower() == 'exit':
                print("Goodbye! 👋")
                break
            
            # Parse commands
            if user_input.startswith('recommend('):
                query = user_input.split('(')[1].split(')')[0].strip('\'"')
                recommend_similar_items(query, top_k=5)
            
            elif user_input.startswith('search('):
                query = user_input.split('(')[1].split(')')[0].strip('\'"')
                results = search_items(query)
                print(f"\nFound {len(results)} items:")
                for idx, row in results.iterrows():
                    print(f"• {row['title']}")
            
            elif user_input.startswith('category('):
                query = user_input.split('(')[1].split(')')[0].strip('\'"')
                recommend_by_category(query)
            
            elif user_input.startswith('compare('):
                parts = user_input.split('(')[1].split(')')[0].split(',')
                if len(parts) == 2:
                    q1 = parts[0].strip('\'" ')
                    q2 = parts[1].strip('\'" ')
                    compare_items(q1, q2)
                else:
                    print("Usage: compare('item1', 'item2')")
            
            elif user_input.startswith('popular()'):
                get_popular_items()
            
            else:
                print("❌ Unknown command. Try: recommend('laptop')")
        
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

# Uncomment to run interactive mode
# interactive_mode()

# ============================================================================
# COMPLETION
# ============================================================================
print("\n" + "=" * 70)
print("✅ RECOMMENDATION SYSTEM IS READY!")
print("=" * 70)

print("\n💡 To use the system:")
print("  recommend_similar_items('product name', top_k=10)")
print("  search_items('keyword')")
print("  recommend_by_category('category')")
print("  compare_items('item1', 'item2')")
print("  get_popular_items()")

print("\n📝 To start interactive mode:")
print("  interactive_mode()")

print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")