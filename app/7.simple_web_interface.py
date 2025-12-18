"""
STEP 6: FLASK API FOR RECOMMENDATION SYSTEM
Simple REST API to serve recommendations
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = r"C:\Users\nagar\Downloads\parquet\processed_data"
VECTORIZED_DIR = os.path.join(DATA_DIR, "vectorized")

# ============================================================================
# LOAD DATA AT STARTUP
# ============================================================================

print("Loading recommendation system...")

all_items_df = pd.read_parquet(os.path.join(VECTORIZED_DIR, 'all_items_processed_100k.parquet'))
item_embeddings = np.load(os.path.join(VECTORIZED_DIR, 'item_embeddings_100k.npy'))

print(f"✓ Loaded {len(all_items_df):,} items")
print(f"✓ Loaded embeddings: {item_embeddings.shape}")

# ============================================================================
# CREATE FLASK APP
# ============================================================================

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ============================================================================
# RECOMMENDATION FUNCTIONS
# ============================================================================

def get_recommendations(query, top_k=10):
    """Get recommendations for a query"""
    # Search for matching items
    mask = all_items_df['title'].str.contains(query, case=False, na=False)
    matches = all_items_df[mask]
    
    if len(matches) == 0:
        return None, "No items found matching your query"
    
    # Get first match
    target_row = matches.iloc[0]
    target_idx = target_row.name
    target_vector = item_embeddings[target_idx].reshape(1, -1)
    
    # Calculate similarities
    sim_scores = np.dot(item_embeddings, target_vector.T).flatten()
    sorted_indices = np.argsort(sim_scores)[::-1]
    
    # Collect recommendations
    recommendations = []
    count = 0
    
    for idx in sorted_indices:
        if idx == target_idx:
            continue
        if count >= top_k:
            break
        
        rec_item = all_items_df.iloc[idx]
        recommendations.append({
            'rank': count + 1,
            'score': float(sim_scores[idx]),
            'asin': rec_item['asin'],
            'title': rec_item['title'],
            'brand': rec_item['brand'],
            'category': rec_item['main_category']
        })
        count += 1
    
    query_item = {
        'asin': target_row['asin'],
        'title': target_row['title'],
        'brand': target_row['brand'],
        'category': target_row['main_category']
    }
    
    return recommendations, query_item

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def home():
    """Home page with simple UI"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Recommendation System</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
            }
            h1 { margin: 0; }
            .search-box {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 30px;
            }
            input {
                width: 70%;
                padding: 15px;
                font-size: 16px;
                border: 2px solid #ddd;
                border-radius: 5px;
            }
            button {
                width: 25%;
                padding: 15px;
                font-size: 16px;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                margin-left: 10px;
            }
            button:hover {
                background: #5568d3;
            }
            .query-item {
                background: #e8f4f8;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 30px;
                border-left: 4px solid #667eea;
            }
            .recommendations {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 20px;
            }
            .rec-item {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }
            .rec-item:hover {
                transform: translateY(-5px);
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            }
            .rank {
                background: #667eea;
                color: white;
                width: 30px;
                height: 30px;
                border-radius: 50%;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                margin-right: 10px;
            }
            .score {
                float: right;
                background: #e8f4f8;
                padding: 5px 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            .title {
                font-weight: bold;
                margin: 10px 0;
                color: #333;
            }
            .meta {
                color: #666;
                font-size: 14px;
            }
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }
            .error {
                background: #fee;
                color: #c33;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 Amazon Product Recommendation System</h1>
            <p>Powered by Two-Tower Neural Network</p>
        </div>

        <div class="search-box">
            <h2>Search for a Product</h2>
            <input type="text" id="query" placeholder="e.g., laptop, headphones, camera..." 
                   onkeypress="if(event.key==='Enter') search()">
            <button onclick="search()">🔍 Get Recommendations</button>
        </div>

        <div class="loading" id="loading">
            <h3>🔄 Finding recommendations...</h3>
        </div>

        <div id="results"></div>

        <script>
            async function search() {
                const query = document.getElementById('query').value;
                if (!query) {
                    alert('Please enter a search query');
                    return;
                }

                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').innerHTML = '';

                try {
                    const response = await fetch(`/api/recommend?q=${encodeURIComponent(query)}&top_k=10`);
                    const data = await response.json();

                    document.getElementById('loading').style.display = 'none';

                    if (data.error) {
                        document.getElementById('results').innerHTML = 
                            `<div class="error">❌ ${data.error}</div>`;
                        return;
                    }

                    let html = '<div class="query-item">';
                    html += `<h3>📦 Your Search: "${query}"</h3>`;
                    html += `<div class="title">${data.query_item.title}</div>`;
                    html += `<div class="meta">Brand: ${data.query_item.brand} | Category: ${data.query_item.category}</div>`;
                    html += '</div>';

                    html += '<h2>✨ Recommended for You</h2>';
                    html += '<div class="recommendations">';

                    data.recommendations.forEach(rec => {
                        html += `
                            <div class="rec-item">
                                <div>
                                    <span class="rank">${rec.rank}</span>
                                    <span class="score">Score: ${rec.score.toFixed(3)}</span>
                                </div>
                                <div class="title">${rec.title}</div>
                                <div class="meta">
                                    🏷️ ${rec.brand}<br>
                                    📁 ${rec.category}
                                </div>
                            </div>
                        `;
                    });

                    html += '</div>';
                    document.getElementById('results').innerHTML = html;

                } catch (error) {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('results').innerHTML = 
                        `<div class="error">❌ Error: ${error.message}</div>`;
                }
            }

            // Sample searches
            const samples = ['laptop', 'headphones', 'camera', 'mouse'];
            document.getElementById('query').value = samples[Math.floor(Math.random() * samples.length)];
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/api/recommend', methods=['GET'])
def recommend():
    """API endpoint for recommendations"""
    query = request.args.get('q', '')
    top_k = int(request.args.get('top_k', 10))
    
    if not query:
        return jsonify({'error': 'Query parameter "q" is required'}), 400
    
    recommendations, query_item = get_recommendations(query, top_k)
    
    if recommendations is None:
        return jsonify({'error': query_item}), 404
    
    return jsonify({
        'query': query,
        'query_item': query_item,
        'recommendations': recommendations,
        'total': len(recommendations)
    })

@app.route('/api/search', methods=['GET'])
def search():
    """API endpoint for searching items"""
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 20))
    
    if not query:
        return jsonify({'error': 'Query parameter "q" is required'}), 400
    
    mask = all_items_df['title'].str.contains(query, case=False, na=False)
    results = all_items_df[mask].head(limit)
    
    items = []
    for _, row in results.iterrows():
        items.append({
            'asin': row['asin'],
            'title': row['title'],
            'brand': row['brand'],
            'category': row['main_category']
        })
    
    return jsonify({
        'query': query,
        'total': len(items),
        'items': items
    })

@app.route('/api/stats', methods=['GET'])
def stats():
    """API endpoint for system statistics"""
    return jsonify({
        'total_items': len(all_items_df),
        'embedding_dim': item_embeddings.shape[1],
        'unique_brands': all_items_df['brand'].nunique(),
        'unique_categories': all_items_df['main_category'].nunique(),
        'top_categories': all_items_df['main_category'].value_counts().head(10).to_dict()
    })

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 RECOMMENDATION API SERVER")
    print("=" * 70)
    print("\nServer starting...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("\n📚 API Endpoints:")
    print("  GET  /                          - Web interface")
    print("  GET  /api/recommend?q=laptop    - Get recommendations")
    print("  GET  /api/search?q=sony         - Search items")
    print("  GET  /api/stats                 - System statistics")
    print("\n⚠️  Press Ctrl+C to stop the server")
    print("=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)