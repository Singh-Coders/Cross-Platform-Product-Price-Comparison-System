import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and prepare data (same as app.py)
df = pd.read_csv('combined_cleaned_data.csv')

# Create searchable column
df['Product_Name_SearchClean'] = df['Product_Name_Extracted'].apply(
    lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)).lower()
)

# Build TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Product_Name_Clean'])

# Test function
def find_cheaper_alternatives(product_name, top_n=5):
    """Find cheaper products similar to the input product"""
    # Clean search term to match database format (remove special characters)
    clean_search = re.sub(r'[^a-zA-Z0-9\s]', '', product_name).lower()
    
    # Search in cleaned column for better matching
    matching_products = df[df['Product_Name_SearchClean'].str.contains(clean_search, case=False, na=False, regex=False)]
    
    if matching_products.empty:
        return None
    
    # Get product index and price
    product_idx = matching_products.index[0]
    original_price = df.loc[product_idx, 'Price_Numeric']
    original_name = df.loc[product_idx, 'Product_Name_Extracted']
    
    # Get TF-IDF vector
    product_vector = tfidf_matrix[product_idx]
    
    # Calculate similarity scores
    similarities = cosine_similarity(product_vector, tfidf_matrix)[0]
    
    # Find cheaper products
    cheaper_mask = df['Price_Numeric'] < original_price
    cheaper_indices = df[cheaper_mask].index.tolist()
    
    if not cheaper_indices:
        return f"No cheaper alternatives found for {original_name}"
    
    # Pair with similarity and sort
    cheaper_similarities = [(idx, similarities[idx]) for idx in cheaper_indices]
    cheaper_similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N
    top_cheaper = cheaper_similarities[:top_n]
    result_indices = [idx for idx, sim in top_cheaper]
    
    # Build result DataFrame
    results = df.loc[result_indices, ['Product_Name_Extracted', 'Price_Numeric', 'Source', 'Type']].copy()
    results['Similarity_Score'] = [sim for idx, sim in top_cheaper]
    results['Original_Price'] = original_price
    results['Price_Difference'] = results['Original_Price'] - results['Price_Numeric']
    results['Savings_Percentage'] = (results['Price_Difference'] / results['Original_Price'] * 100).round(2)
    
    return results, original_name

# Test the search
print("Testing updated search with: 'Samsung 55 4K TV'")
print("="*70)
result = find_cheaper_alternatives('Samsung 55 4K TV', top_n=5)

if result is None:
    print("❌ Product not found")
elif isinstance(result, str):
    print(f"ℹ️ {result}")
else:
    results, original_name = result
    print(f"✅ FOUND: {original_name}")
    print(f"Original Price: ₹{results.iloc[0]['Original_Price']:.0f}\n")
    print("Cheaper Alternatives by Similarity:")
    print(results[['Product_Name_Extracted', 'Price_Numeric', 'Similarity_Score', 'Savings_Percentage']].to_string())
