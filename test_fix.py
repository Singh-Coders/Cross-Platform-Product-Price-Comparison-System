import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and prepare data
df = pd.read_csv('combined_cleaned_data.csv')

# Build TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Product_Name_Clean'])

# Test the improved search function
def find_cheaper_alternatives(product_name, top_n=5):
    """Find cheaper products similar to the input product"""
    # Clean search term (remove special characters for better matching)
    clean_search = re.sub(r'[^a-zA-Z0-9\s]', '', product_name)
    
    # Search for product (case-insensitive, fuzzy matching)
    matching_products = df[
        df['Product_Name_Extracted'].str.contains(product_name, case=False, na=False, regex=False) |
        df['Product_Name_Extracted'].str.contains(clean_search, case=False, na=False, regex=False)
    ]
    
    if matching_products.empty:
        return f"❌ Product '{product_name}' not found"
    
    # Get product index and price
    product_idx = matching_products.index[0]
    original_price = df.loc[product_idx, 'Price_Numeric']
    
    # Get TF-IDF vector
    product_vector = tfidf_matrix[product_idx]
    
    # Calculate similarity scores
    similarities = cosine_similarity(product_vector, tfidf_matrix)[0]
    
    # Find cheaper products
    cheaper_mask = df['Price_Numeric'] < original_price
    cheaper_indices = df[cheaper_mask].index.tolist()
    
    if not cheaper_indices:
        return f"ℹ️ No cheaper alternatives found for {df.loc[product_idx, 'Product_Name_Extracted']}"
    
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
    
    return results

# Test the search
print("Testing search with: 'Samsung 55 4K TV'")
print("="*60)
result = find_cheaper_alternatives('Samsung 55 4K TV', top_n=3)

if isinstance(result, str):
    print(result)
else:
    print("✅ Found! Cheaper alternatives:")
    print(result[['Product_Name_Extracted', 'Price_Numeric', 'Savings_Percentage']])
