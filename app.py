import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
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

CATEGORY_ALIASES = {
    "Air Fryer": ["air fryer"],
    "Book": ["book", "novel"],
    "Book Set": ["book set", "box set"],
    "Body Wash": ["body wash", "shower gel"],
    "Camera": ["camera", "dslr", "mirrorless"],
    "Dishwasher": ["dishwasher"],
    "Drone": ["drone"],
    "Epilator": ["epilator"],
    "Headphones": ["headphone", "headphones", "earphone", "earphones", "earbud", "earbuds", "tws", "buds"],
    "Jacket": ["jacket", "coat"],
    "Jeans": ["jeans", "denim"],
    "Kurta": ["kurta"],
    "Laptop": ["laptop", "notebook", "macbook", "victus", "thinkpad", "ideapad"],
    "Lipstick": ["lipstick"],
    "Microwave": ["microwave", "oven"],
    "Mobile": ["mobile", "phone", "smartphone", "iphone", "galaxy", "redmi", "iqoo", "oneplus", "realme"],
    "Pressure Cooker": ["pressure cooker", "cooker"],
    "Refrigerator": ["refrigerator", "fridge"],
    "Shaving": ["shaver", "trimmer", "razor", "shaving"],
    "Shirt": ["shirt"],
    "Shoes": ["shoe", "shoes", "sneaker", "sneakers", "stan smith"],
    "Skincare": ["skincare", "serum", "moisturizer", "cleanser"],
    "Smartphone": ["mobile", "phone", "smartphone", "iphone", "galaxy", "redmi", "iqoo", "oneplus", "realme"],
    "Smartwatch": ["smartwatch", "watch"],
    "Soap": ["soap"],
    "Sunscreen": ["sunscreen", "spf"],
    "Tablet": ["tablet", "ipad", "tab"],
    "Television": ["tv", "television", "smart tv", "4k tv", "led tv"],
    "Toothbrush": ["toothbrush"],
    "T-Shirt": ["tshirt", "t shirt", "tee"],
    "Vacuum Cleaner": ["vacuum cleaner", "vacuum"],
    "Washing Machine": ["washing machine", "washer", "front load", "top load"],
    "Water Purifier": ["water purifier", "purifier", "ro"],
}

TYPE_GROUPS = {
    "Mobile": {"Mobile", "Smartphone"},
    "Smartphone": {"Mobile", "Smartphone"},
}

TOKEN_STOPWORDS = {
    "a", "an", "and", "ai", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with",
    "black", "blue", "gold", "green", "grey", "gray", "purple", "red", "silver", "white",
}

MIN_RECOMMENDATION_SCORE = 0.35
MIN_RECOMMENDATION_TOKEN_OVERLAP = 0.50


def clean_text(text):
    """
    Consistent text cleaning function used throughout the app.
    Removes special characters, normalizes whitespace, converts to lowercase.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = text.replace("&", " and ").replace("+", " plus ")
    text = re.sub(r'(\d+(?:\.\d+)?)\s*(?:\"|inch|inches)\b', r'\1 inch', text)
    text = re.sub(r'(\d+(?:\.\d+)?)\s*(gb|tb|mah|mp|hz|kg|w)\b', r'\1 \2', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()


def get_meaningful_tokens(text):
    """Return useful query/product tokens after lightweight normalization."""
    return [
        token for token in clean_text(text).split()
        if len(token) > 1 and token not in TOKEN_STOPWORDS
    ]


def infer_query_types(clean_search):
    """Infer product category from query words so matches stay in the right catalog area."""
    matches = set()
    padded_search = f" {clean_search} "
    for product_type, aliases in CATEGORY_ALIASES.items():
        for alias in aliases:
            clean_alias = clean_text(alias)
            if f" {clean_alias} " in padded_search:
                matches.add(product_type)
                break
    return matches


def compatible_types(product_type):
    """Allow equivalent catalog categories to compare with each other."""
    return TYPE_GROUPS.get(product_type, {product_type})


def fuzzy_match_score(s1, s2):
    """
    Calculate fuzzy match score between two strings (0 to 1).
    Higher score = better match.
    """
    return SequenceMatcher(None, s1, s2).ratio()


def token_coverage_score(query_tokens, product_text):
    """Score how many important query tokens are present in a product."""
    if not query_tokens:
        return 0
    product_tokens = set(get_meaningful_tokens(product_text))
    matches = sum(1 for token in query_tokens if token in product_tokens)
    return matches / len(query_tokens)


def numeric_tokens(text):
    """Keep model/size numbers that usually define a specific product variant."""
    return {
        token for token in get_meaningful_tokens(text)
        if any(char.isdigit() for char in token)
    }


def is_reasonable_alternative(original_text, candidate_text, similarity):
    """Reject category-only matches that are not actually close to the selected product."""
    original_tokens = get_meaningful_tokens(original_text)
    token_overlap = token_coverage_score(original_tokens, candidate_text)

    original_numbers = numeric_tokens(original_text)
    candidate_numbers = numeric_tokens(candidate_text)
    if original_numbers and candidate_numbers and original_numbers.isdisjoint(candidate_numbers):
        return False, token_overlap, 0

    recommendation_score = (0.70 * similarity) + (0.30 * token_overlap)
    is_close_enough = (
        recommendation_score >= MIN_RECOMMENDATION_SCORE
        and token_overlap >= MIN_RECOMMENDATION_TOKEN_OVERLAP
    )
    return is_close_enough, token_overlap, recommendation_score


def get_candidate_indices(search_term, df):
    """Use category hints first; fall back to the full dataframe when no hint exists."""
    clean_search = clean_text(search_term)
    query_types = infer_query_types(clean_search)

    if not query_types:
        return df.index.tolist(), query_types

    expanded_types = set()
    for product_type in query_types:
        expanded_types.update(compatible_types(product_type))

    candidates = df[df['Type'].isin(expanded_types)]
    if candidates.empty:
        return df.index.tolist(), query_types
    return candidates.index.tolist(), query_types


def find_best_product_match(search_term, df, threshold=0.32):
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
    
    candidate_indices, query_types = get_candidate_indices(clean_search, df)
    candidate_df = df.loc[candidate_indices]
    query_tokens = get_meaningful_tokens(clean_search)

    # -------- STRATEGY 1: Exact Substring Match --------
    exact_matches = candidate_df[candidate_df['Product_SearchText'].str.contains(clean_search, case=False, na=False, regex=False)]
    if not exact_matches.empty:
        exact_matches = exact_matches.copy()
        exact_matches['_rank'] = exact_matches['Product_SearchText'].apply(
            lambda text: token_coverage_score(query_tokens, text) - (len(text) / 10000)
        )
        return exact_matches['_rank'].idxmax(), 1.0, "exact_match"
    
    # -------- STRATEGY 2: Category-aware token + TF-IDF ranking --------
    try:
        search_vector = tfidf_vectorizer.transform([clean_search])
        similarities = cosine_similarity(search_vector, tfidf_matrix[candidate_indices])[0]

        ranked = []
        for position, idx in enumerate(candidate_indices):
            product_text = df.loc[idx, 'Product_SearchText']
            coverage = token_coverage_score(query_tokens, product_text)
            fuzzy = fuzzy_match_score(" ".join(sorted(query_tokens)), " ".join(sorted(get_meaningful_tokens(product_text))))
            score = (0.55 * similarities[position]) + (0.35 * coverage) + (0.10 * fuzzy)
            ranked.append((idx, score, similarities[position], coverage, fuzzy))

        ranked.sort(key=lambda item: item[1], reverse=True)
        best_idx, best_score, best_tfidf, best_coverage, best_fuzzy = ranked[0]

        if best_score >= threshold or (query_types and best_coverage >= 0.45):
            method = "tfidf_match" if best_tfidf >= best_fuzzy else "fuzzy_match"
            return best_idx, best_score, method
    except Exception:
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
    df = df.reset_index(drop=True)
    
    # Create consistent search column
    df['Product_Name_SearchClean'] = df['Product_Name_Extracted'].apply(clean_text)
    df['Product_SearchText'] = (
        df['Product_Name_Extracted'].fillna('').apply(clean_text)
        + ' '
        + df['Product_Name_Clean'].fillna('').apply(clean_text)
        + ' '
        + df['Type'].fillna('').apply(clean_text)
    )
    
    # Build TF-IDF matrix once
    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Product_SearchText'].fillna(''))
    
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
    original_type = df.loc[product_idx, 'Type']
    
    # Get TF-IDF vector for the matched product
    product_vector = tfidf_matrix[product_idx]
    
    # Calculate similarity scores with ALL products
    similarities = cosine_similarity(product_vector, tfidf_matrix)[0]
    
    # Find cheaper products in the same product family
    same_type_mask = df['Type'].isin(compatible_types(original_type))
    cheaper_mask = (df['Price_Numeric'] < original_price) & same_type_mask & (df.index != product_idx)
    cheaper_indices = df[cheaper_mask].index.tolist()
    
    if not cheaper_indices:
        # No cheaper products found
        return None, product_idx, "no_cheaper"
    
    # Pair with similarity and sort by similarity
    original_text = df.loc[product_idx, 'Product_SearchText']
    cheaper_similarities = []
    for idx in cheaper_indices:
        candidate_text = df.loc[idx, 'Product_SearchText']
        similarity = similarities[idx]
        is_close_enough, token_overlap, recommendation_score = is_reasonable_alternative(
            original_text,
            candidate_text,
            similarity
        )
        if is_close_enough:
            cheaper_similarities.append((idx, similarity, token_overlap, recommendation_score))

    if not cheaper_similarities:
        return None, product_idx, "no_cheaper"

    cheaper_similarities.sort(key=lambda x: x[3], reverse=True)
    
    # Get top N
    top_cheaper = cheaper_similarities[:top_n]
    result_indices = [idx for idx, sim, overlap, score in top_cheaper]
    
    # Build results
    results = df.loc[result_indices, ['Product_Name_Extracted', 'Price_Numeric', 'Source', 'Type']].copy()
    results['Similarity_Score'] = [sim for idx, sim, overlap, score in top_cheaper]
    results['Original_Price'] = original_price
    results['Price_Difference'] = results['Original_Price'] - results['Price_Numeric']
    results['Savings_Percentage'] = (results['Price_Difference'] / results['Original_Price'] * 100).round(2)
    
    return results, product_idx, match_method


def get_similar_product_suggestions(search_term, top_n=5):
    """
    Get suggestions of similar product names when no match is found.
    """
    clean_search = clean_text(search_term)
    
    candidate_indices, _ = get_candidate_indices(clean_search, df)
    query_tokens = get_meaningful_tokens(clean_search)
    fuzzy_scores = []
    for idx in candidate_indices:
        product_text = df.loc[idx, 'Product_SearchText']
        fuzzy = fuzzy_match_score(clean_search, product_text)
        coverage = token_coverage_score(query_tokens, product_text)
        score = (0.6 * coverage) + (0.4 * fuzzy)
        fuzzy_scores.append((df.loc[idx, 'Product_Name_Extracted'], score))
    
    fuzzy_scores.sort(key=lambda x: x[1], reverse=True)
    return fuzzy_scores[:top_n]


# ============================================================================
# DATASET OVERVIEW FUNCTIONS
# ============================================================================

def get_dataset_statistics(df):
    """Generate comprehensive dataset statistics"""
    stats = {
        'total_products': len(df),
        'unique_products': df['Product_Name_Extracted'].nunique(),
        'price_min': df['Price_Numeric'].min(),
        'price_max': df['Price_Numeric'].max(),
        'price_mean': df['Price_Numeric'].mean(),
        'price_median': df['Price_Numeric'].median(),
        'price_std': df['Price_Numeric'].std(),
        'categories': df['Type'].nunique(),
        'sources': df['Source'].nunique(),
    }
    return stats


def get_category_distribution(df):
    """Get distribution of products by category"""
    return df['Type'].value_counts()


def get_source_distribution(df):
    """Get distribution of products by source (Amazon, Flipkart, etc.)"""
    return df['Source'].value_counts()


def get_price_by_category(df):
    """Get average price by category"""
    return df.groupby('Type')['Price_Numeric'].agg(['mean', 'min', 'max', 'count']).round(0)


def get_products_with_cheaper_alternatives(df, tfidf_matrix, min_similarity=0.3):
    """
    Find all products that have cheaper alternatives available.
    Returns a dataframe with product info and details about their cheaper alternatives.
    """
    products_with_alternatives = []
    
    for idx, row in df.iterrows():
        original_price = row['Price_Numeric']
        original_type = row['Type']
        
        # Get similar products using TF-IDF
        product_vector = tfidf_matrix[idx]
        similarities = cosine_similarity(product_vector, tfidf_matrix)[0]
        
        # Find cheaper products in same category
        same_type_mask = df['Type'].isin(compatible_types(original_type))
        cheaper_mask = (df['Price_Numeric'] < original_price) & same_type_mask & (df.index != idx)
        
        # Filter by minimum similarity
        cheaper_candidates = df[cheaper_mask]
        if not cheaper_candidates.empty:
            cheaper_with_sim = []
            original_text = row['Product_SearchText']
            for cidx in cheaper_candidates.index:
                candidate_text = df.loc[cidx, 'Product_SearchText']
                is_close_enough, _, recommendation_score = is_reasonable_alternative(
                    original_text,
                    candidate_text,
                    similarities[cidx]
                )
                if similarities[cidx] >= min_similarity and is_close_enough:
                    cheaper_with_sim.append(cidx)
            
            if cheaper_with_sim:
                # Get the best (most similar) cheaper alternative
                best_cheaper_idx = max(
                    cheaper_with_sim,
                    key=lambda x: is_reasonable_alternative(
                        original_text,
                        df.loc[x, 'Product_SearchText'],
                        similarities[x]
                    )[2]
                )
                best_cheaper = df.loc[best_cheaper_idx]
                
                # Calculate savings
                savings_amount = original_price - best_cheaper['Price_Numeric']
                savings_percent = (savings_amount / original_price) * 100
                
                products_with_alternatives.append({
                    'Product_Name': row['Product_Name_Extracted'],
                    'Category': row['Type'],
                    'Original_Price': original_price,
                    'Cheaper_Alternative': best_cheaper['Product_Name_Extracted'],
                    'Alternative_Price': best_cheaper['Price_Numeric'],
                    'Savings_Amount': savings_amount,
                    'Savings_Percent': savings_percent,
                    'Source': row['Source'],
                    'Num_Alternatives': len(cheaper_with_sim)
                })
    
    result_df = pd.DataFrame(products_with_alternatives)
    # Return 10 random products from those with cheaper alternatives
    if len(result_df) > 10:
        return result_df.sample(n=10, random_state=None).reset_index(drop=True)
    return result_df.reset_index(drop=True)


def display_dataset_overview():
    """Display comprehensive dataset overview in Streamlit"""
    st.header("📊 Dataset Overview")
    
    # Get statistics
    stats = get_dataset_statistics(df)
    
    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Products", f"{stats['total_products']:,}")
    with col2:
        st.metric("Unique Products", f"{stats['unique_products']:,}")
    with col3:
        st.metric("Categories", f"{stats['categories']}")
    with col4:
        st.metric("Sources", f"{stats['sources']}")
    with col5:
        st.metric("Avg Price", f"₹{stats['price_mean']:,.0f}")
    
    st.divider()
    
    # Price statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("💰 Price Statistics")
        st.write(f"**Minimum:** ₹{stats['price_min']:,.0f}")
        st.write(f"**Maximum:** ₹{stats['price_max']:,.0f}")
        st.write(f"**Median:** ₹{stats['price_median']:,.0f}")
        st.write(f"**Std Dev:** ₹{stats['price_std']:,.0f}")
    
    with col2:
        st.subheader("📦 Category Distribution")
        category_dist = get_category_distribution(df)
        st.bar_chart(category_dist)
    
    with col3:
        st.subheader("🏪 Source Distribution")
        source_dist = get_source_distribution(df)
        st.bar_chart(source_dist)
    
    st.divider()
    
    # Detailed category analysis
    st.subheader("🔍 Detailed Category Analysis")
    price_by_cat = get_price_by_category(df)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Category Performance:**")
        st.dataframe(price_by_cat, use_container_width=True)
    
    with col2:
        st.write("**Average Price by Category:**")
        fig, ax = plt.subplots(figsize=(10, 6))
        price_by_cat['mean'].sort_values(ascending=False).plot(
            kind='barh', ax=ax, color='steelblue'
        )
        ax.set_xlabel('Average Price (₹)')
        ax.set_title('Average Price by Product Category')
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig, use_container_width=True)
    
    st.divider()
    
    # Products with cheaper alternatives
    st.subheader("🎯 Products with Cheaper Alternatives Available")
    
    with st.spinner("Finding 10 random products with cheaper alternatives..."):
        alternatives_df = get_products_with_cheaper_alternatives(df, tfidf_matrix, min_similarity=0.3)
    
    if not alternatives_df.empty:
        # Display statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sample Size", len(alternatives_df))
        with col2:
            st.metric("Avg Savings %", f"{alternatives_df['Savings_Percent'].mean():.1f}%")
        with col3:
            st.metric("Max Savings %", f"{alternatives_df['Savings_Percent'].max():.1f}%")
        
        st.write("**Random sample of products with cheaper alternatives:**")
        
        # Format for display
        display_alts = alternatives_df.copy()
        display_alts['Original_Price'] = display_alts['Original_Price'].apply(lambda x: f"₹{x:,.0f}")
        display_alts['Alternative_Price'] = display_alts['Alternative_Price'].apply(lambda x: f"₹{x:,.0f}")
        display_alts['Savings_Amount'] = display_alts['Savings_Amount'].apply(lambda x: f"₹{x:,.0f}")
        display_alts['Savings_Percent'] = display_alts['Savings_Percent'].apply(lambda x: f"{x:.1f}%")
        
        display_alts.columns = [
            'Product', 'Category', 'Original Price', 'Cheaper Alternative', 
            'Alternative Price', 'Savings Amount', 'Savings %', 'Source', '# of Alternatives'
        ]
        
        st.dataframe(
            display_alts[['Product', 'Category', 'Original Price', 'Cheaper Alternative', 
                         'Alternative Price', 'Savings Amount', 'Savings %', '# of Alternatives']].reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No products with cheaper alternatives found in the current dataset.")
    
    st.divider()
    
    # Source-wise breakdown
    st.subheader("🔗 Source-wise Breakdown")
    source_stats = df.groupby('Source').agg({
        'Product_Name_Extracted': 'count',
        'Price_Numeric': ['mean', 'min', 'max']
    }).round(0)
    source_stats.columns = ['Total Products', 'Avg Price', 'Min Price', 'Max Price']
    st.dataframe(source_stats, use_container_width=True)


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    # Header
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1>🛍️ Product Alternative Finder</h1>
            <p style='font-size: 18px; color: gray;'>Find cheaper products similar to your favorite items</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Main navigation tabs
    main_tabs = st.tabs(["🔍 Search", "📊 Dataset Overview"])
    
    with main_tabs[0]:  # Search Tab
        # Sidebar for settings
        with st.sidebar:
            st.header("⚙️ Settings")
            
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
            st.subheader("Quick Examples (10-20% Savings)")
            example_products = ['Cinthol Deodorant Soap', 'JBL Flip 6', 'Adidas NMD R1']
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
                    results, product_idx, match_method = find_cheaper_alternatives(search_input, top_n=top_n)
                
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
                    # Matched a product but no alternatives found
                    original_product = df.loc[product_idx]
                    st.info(f"✅ Found: {original_product['Product_Name_Extracted']} (₹{original_product['Price_Numeric']:,.0f})")
                    
                    if match_method == "no_cheaper":
                        st.warning(f"⚠️ No cheaper alternatives found for this product")
                
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
                    
                    # Display cheaper alternatives results
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
                    
                    st.dataframe(
                        display_results.reset_index(drop=True),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Stats
                    if len(results) > 0:
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
    
    with main_tabs[1]:  # Dataset Overview Tab
        display_dataset_overview()
    
    st.divider()
    
    # Footer info
    st.markdown("""
        <div style='text-align: center; color: gray; font-size: 12px; padding: 20px;'>
            <p>Model: TF-IDF + Cosine Similarity + Fuzzy Matching | Database: {0} products | Last Updated: April 2026</p>
        </div>
    """.format(len(df)), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
