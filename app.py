# import streamlit as st
# import pandas as pd
# import re
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # ============================================================================
# # PAGE CONFIGURATION
# # ============================================================================
# st.set_page_config(
#     page_title="Product Alternative Finder",
#     page_icon="🛍️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ============================================================================
# # CACHE DATA LOADING
# # ============================================================================
# @st.cache_data
# def load_and_prepare_data():
#     """Load data and prepare TF-IDF matrix"""
#     # Load CSV (using combined dataset with more products)
#     df = pd.read_csv('combined_cleaned_data.csv')
    
#     # Data is already cleaned, columns ready to use:
#     # - Product_Name_Extracted: clean product name
#     # - Product_Name_Clean: text cleaned for TF-IDF
#     # - Price_Numeric: numeric price for comparison
#     # - Type: product category
#     # - Source: Amazon or Flipkart
    
#     # Create a searchable column (remove special characters for better matching)
#     df['Product_Name_SearchClean'] = df['Product_Name_Extracted'].apply(
#         lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)).lower()
#     )
    
#     # Build TF-IDF matrix
#     tfidf_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
#     tfidf_matrix = tfidf_vectorizer.fit_transform(df['Product_Name_Clean'])
    
#     return df, tfidf_matrix

# # Load data
# df, tfidf_matrix = load_and_prepare_data()

# # ============================================================================
# # MODEL FUNCTION
# # ============================================================================
# def find_cheaper_alternatives(product_name, top_n=5):
#     """
#     Find cheaper products similar to the input product.
    
#     Parameters:
#     -----------
#     product_name : str
#         The product name to search for
#     top_n : int
#         Number of cheaper alternatives to return
    
#     Returns:
#     --------
#     DataFrame or error message
#     """
#     # Clean search term to match database format (remove special characters)
#     clean_search = re.sub(r'[^a-zA-Z0-9\s]', '', product_name).lower()
    
#     # Search in cleaned column for better matching
#     matching_products = df[df['Product_Name_SearchClean'].str.contains(clean_search, case=False, na=False, regex=False)]
    
#     if matching_products.empty:
#         return None
    
#     # Get product index and price
#     product_idx = matching_products.index[0]
#     original_price = df.loc[product_idx, 'Price_Numeric']
    
#     # Get TF-IDF vector
#     product_vector = tfidf_matrix[product_idx]
    
#     # Calculate similarity scores
#     similarities = cosine_similarity(product_vector, tfidf_matrix)[0]
    
#     # Find cheaper products
#     cheaper_mask = df['Price_Numeric'] < original_price
#     cheaper_indices = df[cheaper_mask].index.tolist()
    
#     # Pair with similarity and sort
#     cheaper_similarities = [(idx, similarities[idx]) for idx in cheaper_indices]
#     cheaper_similarities.sort(key=lambda x: x[1], reverse=True)
    
#     # Get top N
#     top_cheaper = cheaper_similarities[:top_n]
#     result_indices = [idx for idx, sim in top_cheaper]
    
#     # Build results
#     results = df.loc[result_indices, ['Product_Name_Extracted', 'Price_Numeric', 'Source', 'Type']].copy()
#     results['Similarity_Score'] = [sim for idx, sim in top_cheaper]
#     results['Original_Price'] = original_price
#     results['Price_Difference'] = results['Original_Price'] - results['Price_Numeric']
#     results['Savings_Percentage'] = (results['Price_Difference'] / results['Original_Price'] * 100).round(2)
    
#     return results

# def find_same_product_variants(product_name, top_n=5):
#     """
#     Find the SAME product from different sources/variants.
    
#     Parameters:
#     -----------
#     product_name : str
#         The product name to search for
#     top_n : int
#         Number of variants to return
    
#     Returns:
#     --------
#     DataFrame or None
#     """
#     # Clean search term
#     clean_search = re.sub(r'[^a-zA-Z0-9\s]', '', product_name).lower().strip()
    
#     # Find exact product matches (same product name)
#     matching_products = df[df['Product_Name_SearchClean'].str.contains(clean_search, case=False, na=False, regex=False)]
    
#     if matching_products.empty:
#         return None
    
#     # Sort by price to show all variants
#     variants = matching_products[['Product_Name_Extracted', 'Price_Numeric', 'Source', 'Type']].sort_values('Price_Numeric').head(top_n)
    
#     if len(variants) == 0:
#         return None
    
#     # Add additional info
#     variants = variants.copy()
#     min_price = variants['Price_Numeric'].min()
#     max_price = variants['Price_Numeric'].max()
    
#     variants['Price_Difference_from_Min'] = variants['Price_Numeric'] - min_price
#     variants['Available_Count'] = len(matching_products)
    
#     return variants

# # ============================================================================
# # STREAMLIT UI
# # ============================================================================

# # Header
# st.markdown("""
#     <div style='text-align: center; padding: 20px;'>
#         <h1>🛍️ Product Alternative Finder</h1>
#         <p style='font-size: 18px; color: gray;'>Find cheaper products similar to your favorite items</p>
#     </div>
# """, unsafe_allow_html=True)

# st.divider()

# # Sidebar for settings
# with st.sidebar:
#     st.header("⚙️ Settings")
    
#     search_mode = st.radio(
#         "Search Mode",
#         options=["Cheaper Alternatives", "Same Product Variants"],
#         help="Find cheaper alternatives OR see all variants of the same product"
#     )
    
#     top_n = st.slider(
#         "Number of results to show",
#         min_value=1,
#         max_value=10,
#         value=5,
#         help="How many results would you like to see?"
#     )
    
#     st.divider()
    
#     st.subheader("📋 Available Products")
#     st.write(f"Total products in database: **{len(df)}**")
#     st.write(f"Unique product names: **{df['Product_Name_Extracted'].nunique()}**")

# # Main content area
# col1, col2 = st.columns([2, 1], gap="large")

# with col1:
#     st.subheader("Search for a Product")
    
#     # Search input
#     search_input = st.text_input(
#         "Enter product name:",
#         placeholder="e.g., Samsung Galaxy S26, iPhone, HP Victus...",
#         label_visibility="collapsed"
#     )
    
#     search_button = st.button("🔍 Find Alternatives", type="primary", use_container_width=True)

# with col2:
#     st.subheader("Quick Examples")
#     example_products = ['HP Victus', 'Samsung Galaxy S26 5G', 'iQOO Z11x 5G']
#     for product in example_products:
#         if st.button(product, use_container_width=True):
#             search_input = product

# # Display results
# if search_button or search_input:
#     if not search_input.strip():
#         st.warning("⚠️ Please enter a product name to search")
#     else:
#         with st.spinner(f"🔍 Searching for '{search_input}'..."):
#             if search_mode == "Cheaper Alternatives":
#                 results = find_cheaper_alternatives(search_input, top_n=top_n)
#             else:  # Same Product Variants
#                 results = find_same_product_variants(search_input, top_n=top_n)
        
#         if results is None:
#             st.error(f"❌ Product '{search_input}' not found in database")
#             st.info("💡 Try searching for a different product name or use one of the quick examples")
#         else:
#             # Original product info
#             original_product = df[df['Product_Name_Extracted'].str.contains(search_input, case=False, na=False)].iloc[0]
            
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 st.metric("Original Product", original_product['Product_Name_Extracted'][:30])
#             with col2:
#                 st.metric("Original Price", f"₹{original_product['Price_Numeric']:,.0f}")
#             with col3:
#                 st.metric("Type", original_product['Type'])
            
#             st.divider()
            
#             # Results based on search mode
#             if search_mode == "Cheaper Alternatives":
#                 st.subheader(f"✨ {len(results)} Cheaper Alternatives Found!")
                
#                 # Format results for display
#                 display_results = results.copy()
#                 display_results['Price_Numeric'] = display_results['Price_Numeric'].apply(lambda x: f"₹{x:,.0f}")
#                 display_results['Original_Price'] = display_results['Original_Price'].apply(lambda x: f"₹{x:,.0f}")
#                 display_results['Price_Difference'] = display_results['Price_Difference'].apply(lambda x: f"₹{x:,.0f}")
#                 display_results['Similarity_Score'] = display_results['Similarity_Score'].apply(lambda x: f"{x:.2%}")
#                 display_results['Savings_Percentage'] = display_results['Savings_Percentage'].apply(lambda x: f"{x:.1f}%")
                
#                 # Rename columns for display
#                 display_results.columns = [
#                     'Product Name', 'Price', 'Source', 'Type', 'Similarity', 
#                     'Original Price', 'Savings (₹)', 'Savings (%)'
#                 ]
#             else:  # Same Product Variants
#                 st.subheader(f"📦 {len(results)} Variants Available!")
                
#                 # Format results for display
#                 display_results = results.copy()
#                 display_results['Price_Numeric'] = display_results['Price_Numeric'].apply(lambda x: f"₹{x:,.0f}")
#                 display_results['Price_Difference_from_Min'] = display_results['Price_Difference_from_Min'].apply(lambda x: f"₹{x:,.0f}")
                
#                 # Rename columns for display
#                 display_results.columns = [
#                     'Product Name', 'Price', 'Source', 'Type', 
#                     'Price Difference from Cheapest', 'Total Available'
#                 ]
            
#             st.dataframe(
#                 display_results.reset_index(drop=True),
#                 use_container_width=True,
#                 hide_index=True
#             )
            
#             # Stats
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 max_savings = results['Savings_Percentage'].max()
#                 st.metric("Max Savings", f"{max_savings:.1f}%")
#             with col2:
#                 avg_savings = results['Savings_Percentage'].mean()
#                 st.metric("Avg Savings", f"{avg_savings:.1f}%")
#             with col3:
#                 max_discount_rupees = results['Price_Difference'].max()
#                 st.metric("Max Savings (₹)", f"₹{max_discount_rupees:,.0f}")

# st.divider()

# # Footer info
# st.markdown("""
#     <div style='text-align: center; color: gray; font-size: 12px; padding: 20px;'>
#         <p>Model: TF-IDF + Cosine Similarity | Database: {0} products | Last Updated: April 2026</p>
#     </div>
# """.format(len(df)), unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher  # Built-in fuzzy matching

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Product Alternative Finder",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clean_text(text):
    """
    Consistent text cleaning function used throughout the app.
    Removes special characters, normalizes whitespace, converts to lowercase.
    """
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.lower()


def fuzzy_match_score(s1, s2):
    """
    Calculate fuzzy match score between two strings (0 to 1).
    Higher score = better match.
    """
    return SequenceMatcher(None, s1, s2).ratio()


def find_best_product_match(search_term, df, threshold=0.6):
    """
    Find the best matching product using multiple strategies.
    
    Strategy 1: Exact substring match (fastest)
    Strategy 2: Fuzzy matching (handles typos and partial matches)
    Strategy 3: TF-IDF similarity (semantic matching)
    
    Returns: (matched_index, confidence_score, match_method)
    """
    clean_search = clean_text(search_term)
    
    if not clean_search:
        return None, 0, "empty"
    
    # -------- STRATEGY 1: Exact Substring Match --------
    exact_matches = df[df['Product_Name_SearchClean'].str.contains(clean_search, case=False, na=False, regex=False)]
    if not exact_matches.empty:
        return exact_matches.index[0], 1.0, "exact_match"
    
    # -------- STRATEGY 2: Fuzzy Matching --------
    fuzzy_scores = []
    for idx, product_name in enumerate(df['Product_Name_SearchClean']):
        if pd.notna(product_name):
            score = fuzzy_match_score(clean_search, product_name)
            fuzzy_scores.append((idx, score))
    
    if fuzzy_scores:
        fuzzy_scores.sort(key=lambda x: x[1], reverse=True)
        best_idx, best_score = fuzzy_scores[0]
        
        if best_score >= threshold:
            return best_idx, best_score, "fuzzy_match"
    
    # -------- STRATEGY 3: TF-IDF Similarity --------
    # Vectorize search term and all products
    try:
        tfidf_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        all_texts = list(df['Product_Name_Clean']) + [clean_search]
        tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)
        
        search_vector = tfidf_matrix[-1]  # Last row is search term
        similarities = cosine_similarity(search_vector, tfidf_matrix[:-1])[0]
        
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]
        
        if best_score >= 0.1:  # Lower threshold for TF-IDF
            return best_idx, best_score, "tfidf_match"
    except:
        pass
    
    return None, 0, "no_match"


# ============================================================================
# CACHE DATA LOADING
# ============================================================================
@st.cache_data
def load_and_prepare_data():
    """Load data and prepare TF-IDF matrix"""
    try:
        df = pd.read_csv('combined_cleaned_data.csv')
    except FileNotFoundError:
        st.error("❌ Data file 'combined_cleaned_data.csv' not found!")
        st.stop()
    
    # Validate required columns
    required_cols = ['Product_Name_Extracted', 'Product_Name_Clean', 'Price_Numeric', 'Type', 'Source']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing columns in data: {missing_cols}")
        st.stop()
    
    # Clean and prepare
    df = df.dropna(subset=['Product_Name_Extracted', 'Price_Numeric'])  # Remove rows with null values
    df['Price_Numeric'] = pd.to_numeric(df['Price_Numeric'], errors='coerce')
    df = df.dropna(subset=['Price_Numeric'])
    
    # Create consistent search column
    df['Product_Name_SearchClean'] = df['Product_Name_Extracted'].apply(clean_text)
    
    # Build TF-IDF matrix once
    tfidf_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Product_Name_Clean'].fillna(''))
    
    return df, tfidf_matrix, tfidf_vectorizer


# Load data
df, tfidf_matrix, tfidf_vectorizer = load_and_prepare_data()

# ============================================================================
# MODEL FUNCTIONS
# ============================================================================

def find_cheaper_alternatives(product_name, top_n=5):
    """
    Find cheaper products similar to the input product.
    Uses multi-strategy matching to find products.
    """
    # Find best matching product
    product_idx, confidence, match_method = find_best_product_match(product_name, df)
    
    if product_idx is None:
        return None, None, "no_match"
    
    original_price = df.loc[product_idx, 'Price_Numeric']
    
    # Get TF-IDF vector for the matched product
    product_vector = tfidf_matrix[product_idx]
    
    # Calculate similarity scores with ALL products
    similarities = cosine_similarity(product_vector, tfidf_matrix)[0]
    
    # Find cheaper products
    cheaper_mask = df['Price_Numeric'] < original_price
    cheaper_indices = df[cheaper_mask].index.tolist()
    
    if not cheaper_indices:
        # No cheaper products found
        return None, product_idx, "no_cheaper"
    
    # Pair with similarity and sort by similarity
    cheaper_similarities = [(idx, similarities[idx]) for idx in cheaper_indices]
    cheaper_similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N
    top_cheaper = cheaper_similarities[:top_n]
    result_indices = [idx for idx, sim in top_cheaper]
    
    # Build results
    results = df.loc[result_indices, ['Product_Name_Extracted', 'Price_Numeric', 'Source', 'Type']].copy()
    results['Similarity_Score'] = [sim for idx, sim in top_cheaper]
    results['Original_Price'] = original_price
    results['Price_Difference'] = results['Original_Price'] - results['Price_Numeric']
    results['Savings_Percentage'] = (results['Price_Difference'] / results['Original_Price'] * 100).round(2)
    
    return results, product_idx, match_method


def find_same_product_variants(product_name, top_n=5):
    """
    Find the SAME product from different sources/variants.
    """
    product_idx, confidence, match_method = find_best_product_match(product_name, df)
    
    if product_idx is None:
        return None, None, "no_match"
    
    # Get the matched product name
    matched_name = df.loc[product_idx, 'Product_Name_SearchClean']
    
    # Find ALL products with similar names (variants)
    matching_products = df[df['Product_Name_SearchClean'].str.contains(matched_name.split()[0] if matched_name else '', case=False, na=False, regex=False)]
    
    if matching_products.empty:
        return None, product_idx, "no_variants"
    
    # Sort by price and get top N
    variants = matching_products[['Product_Name_Extracted', 'Price_Numeric', 'Source', 'Type']].sort_values('Price_Numeric').head(top_n)
    
    if len(variants) == 0:
        return None, product_idx, "no_variants"
    
    # Add additional info
    variants = variants.copy()
    min_price = variants['Price_Numeric'].min()
    max_price = variants['Price_Numeric'].max()
    
    variants['Price_Difference_from_Min'] = variants['Price_Numeric'] - min_price
    variants['Available_Count'] = len(matching_products)
    
    return variants, product_idx, match_method


def get_similar_product_suggestions(search_term, top_n=5):
    """
    Get suggestions of similar product names when no match is found.
    """
    clean_search = clean_text(search_term)
    
    fuzzy_scores = []
    for idx, product_name in enumerate(df['Product_Name_SearchClean']):
        if pd.notna(product_name):
            score = fuzzy_match_score(clean_search, product_name)
            fuzzy_scores.append((df.loc[idx, 'Product_Name_Extracted'], score))
    
    fuzzy_scores.sort(key=lambda x: x[1], reverse=True)
    return fuzzy_scores[:top_n]


# ============================================================================
# STREAMLIT UI
# ============================================================================

# Header
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>🛍️ Product Alternative Finder</h1>
        <p style='font-size: 18px; color: gray;'>Find cheaper products similar to your favorite items</p>
    </div>
""", unsafe_allow_html=True)

st.divider()

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    search_mode = st.radio(
        "Search Mode",
        options=["Cheaper Alternatives", "Same Product Variants"],
        help="Find cheaper alternatives OR see all variants of the same product"
    )
    
    top_n = st.slider(
        "Number of results to show",
        min_value=1,
        max_value=10,
        value=5,
        help="How many results would you like to see?"
    )
    
    st.divider()
    
    st.subheader("📋 Database Info")
    st.write(f"Total products: **{len(df)}**")
    st.write(f"Unique products: **{df['Product_Name_Extracted'].nunique()}**")
    st.write(f"Price range: **₹{df['Price_Numeric'].min():,.0f}** - **₹{df['Price_Numeric'].max():,.0f}**")
    
    # Debug info (optional)
    if st.checkbox("Show debug info"):
        st.write("**Search Methods Used:** Exact → Fuzzy → TF-IDF")
        st.write("**Fuzzy Match Threshold:** 60%")

# Main content area
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("Search for a Product")
    
    # Search input
    search_input = st.text_input(
        "Enter product name:",
        placeholder="e.g., Samsung Galaxy S26, iPhone, HP Victus...",
        label_visibility="collapsed"
    )
    
    search_button = st.button("🔍 Find Alternatives", type="primary", use_container_width=True)

with col2:
    st.subheader("Quick Examples")
    example_products = ['HP Victus', 'Samsung Galaxy S26 5G', 'iQOO Z11x 5G']
    for product in example_products:
        if st.button(product, use_container_width=True):
            search_input = product
            search_button = True

# Display results
if search_button or search_input:
    if not search_input.strip():
        st.warning("⚠️ Please enter a product name to search")
    else:
        with st.spinner(f"🔍 Searching for '{search_input}'..."):
            if search_mode == "Cheaper Alternatives":
                results, product_idx, match_method = find_cheaper_alternatives(search_input, top_n=top_n)
            else:  # Same Product Variants
                results, product_idx, match_method = find_same_product_variants(search_input, top_n=top_n)
        
        # Handle different result cases
        if product_idx is None:
            st.error(f"❌ Product '{search_input}' not found in database")
            
            # Suggest similar products
            st.subheader("💡 Did you mean one of these?")
            suggestions = get_similar_product_suggestions(search_input, top_n=5)
            
            col1, col2, col3 = st.columns(3)
            for i, (product_name, score) in enumerate(suggestions[:3]):
                with col1 if i % 3 == 0 else (col2 if i % 3 == 1 else col3):
                    if st.button(f"🔎 {product_name[:40]}", key=f"suggest_{i}"):
                        st.rerun()
            
            st.info("💭 Try using one of the suggested product names above")
        
        elif results is None:
            # Matched a product but no alternatives/variants found
            original_product = df.loc[product_idx]
            st.info(f"✅ Found: {original_product['Product_Name_Extracted']} (₹{original_product['Price_Numeric']:,.0f})")
            
            if match_method == "no_cheaper":
                st.warning(f"⚠️ No cheaper alternatives found for this product")
            elif match_method == "no_variants":
                st.warning(f"⚠️ No other variants found for this product")
        
        else:
            # Results found!
            original_product = df.loc[product_idx]
            
            # Match method indicator
            method_icons = {
                "exact_match": "✅ Exact match",
                "fuzzy_match": "🔍 Fuzzy match (typo correction applied)",
                "tfidf_match": "🧠 Semantic match",
                "no_match": "❌ No match"
            }
            st.success(f"{method_icons.get(match_method, 'Match found')}")
            
            # Original product info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Product", original_product['Product_Name_Extracted'][:40])
            with col2:
                st.metric("Original Price", f"₹{original_product['Price_Numeric']:,.0f}")
            with col3:
                st.metric("Type", original_product['Type'])
            
            st.divider()
            
            # Results based on search mode
            if search_mode == "Cheaper Alternatives":
                st.subheader(f"✨ {len(results)} Cheaper Alternatives Found!")
                
                # Format results for display
                display_results = results.copy()
                display_results['Price_Numeric'] = display_results['Price_Numeric'].apply(lambda x: f"₹{x:,.0f}")
                display_results['Original_Price'] = display_results['Original_Price'].apply(lambda x: f"₹{x:,.0f}")
                display_results['Price_Difference'] = display_results['Price_Difference'].apply(lambda x: f"₹{x:,.0f}")
                display_results['Similarity_Score'] = display_results['Similarity_Score'].apply(lambda x: f"{x:.2%}")
                display_results['Savings_Percentage'] = display_results['Savings_Percentage'].apply(lambda x: f"{x:.1f}%")
                
                # Rename columns for display
                display_results.columns = [
                    'Product Name', 'Price', 'Source', 'Type', 'Similarity', 
                    'Original Price', 'Savings (₹)', 'Savings (%)'
                ]
            else:  # Same Product Variants
                st.subheader(f"📦 {len(results)} Variants Available!")
                
                # Format results for display
                display_results = results.copy()
                display_results['Price_Numeric'] = display_results['Price_Numeric'].apply(lambda x: f"₹{x:,.0f}")
                display_results['Price_Difference_from_Min'] = display_results['Price_Difference_from_Min'].apply(lambda x: f"₹{x:,.0f}")
                
                # Rename columns for display
                display_results.columns = [
                    'Product Name', 'Price', 'Source', 'Type', 
                    'Price Difference from Cheapest', 'Total Available'
                ]
            
            st.dataframe(
                display_results.reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )
            
            # Stats (only for cheaper alternatives)
            if search_mode == "Cheaper Alternatives" and len(results) > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    max_savings = results['Savings_Percentage'].max()
                    st.metric("Max Savings", f"{max_savings:.1f}%")
                with col2:
                    avg_savings = results['Savings_Percentage'].mean()
                    st.metric("Avg Savings", f"{avg_savings:.1f}%")
                with col3:
                    max_discount_rupees = results['Price_Difference'].max()
                    st.metric("Max Savings (₹)", f"₹{max_discount_rupees:,.0f}")

st.divider()

# Footer info
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px; padding: 20px;'>
        <p>Model: TF-IDF + Cosine Similarity + Fuzzy Matching | Database: {0} products | Last Updated: April 2026</p>
    </div>
""".format(len(df)), unsafe_allow_html=True)
