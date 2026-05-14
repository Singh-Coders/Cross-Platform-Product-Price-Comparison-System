# Product Alternative Finder

## Project Report

### 1. Title

**Product Alternative Finder using TF-IDF, Cosine Similarity, and Fuzzy Matching**

### 2. Abstract

The Product Alternative Finder is a Python-based web application that helps users find cheaper products similar to a selected product. The project uses product data collected from Amazon and Flipkart, cleans and combines the dataset, and applies text-based similarity techniques to recommend alternatives. The application is built using Streamlit for the user interface and Scikit-learn for machine learning-based text vectorization and similarity comparison.

The system allows users to search for a product, identify the closest matching product in the database, and view cheaper alternatives ranked by similarity. It also supports same-product variant comparison across platforms. This makes the project useful for price-aware online shopping and product comparison.

### 3. Introduction

Online shopping platforms contain thousands of products with different prices, specifications, and sellers. A customer often spends a lot of time manually comparing similar products across platforms such as Amazon and Flipkart. This project solves that problem by building a recommendation system that searches a product database and suggests cheaper alternatives based on product-name similarity.

The main idea is to convert product names into numerical vectors using TF-IDF and compare those vectors using cosine similarity. The system also uses exact matching and fuzzy matching so that the search still works even if the user enters a partial name or makes a small typing mistake.

### 4. Problem Statement

Customers need an easier way to find similar products at lower prices across e-commerce platforms. Manual comparison is time-consuming and can miss better deals. The project aims to create a system that automatically finds cheaper alternatives for a searched product and displays the expected savings.

### 5. Objectives

- To collect and prepare product data from Amazon and Flipkart.
- To clean product names and prices for reliable processing.
- To build a searchable product database.
- To apply TF-IDF vectorization for product-name representation.
- To use cosine similarity for ranking similar products.
- To recommend only products that are cheaper than the selected product.
- To provide a simple Streamlit web interface for users.
- To show savings in rupees and percentage.

### 6. Existing System

In the existing manual approach, users search products separately on different e-commerce websites and compare names, specifications, and prices by themselves. This approach has the following drawbacks:

- It takes more time.
- Similar products may be difficult to identify manually.
- Users may overlook cheaper alternatives.
- There is no automatic similarity scoring.
- Comparing products across platforms is inconvenient.

### 7. Proposed System

The proposed system provides an automated product alternative finder. The user enters a product name, and the system searches the combined product dataset. After finding the best matching product, it calculates similarity with all other products and filters products that have a lower price. The final results are sorted by similarity score and displayed with savings.

The system includes two search modes:

- **Cheaper Alternatives:** Finds similar products with lower prices.
- **Same Product Variants:** Shows available variants of the same or similar product sorted by price.

### 8. Technology Stack

| Component | Technology Used |
|---|---|
| Programming Language | Python |
| Web Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Text Representation | TF-IDF Vectorizer |
| Similarity Measurement | Cosine Similarity |
| Fuzzy Search | Python `difflib.SequenceMatcher` |
| Data Format | CSV |
| Development Environment | Jupyter Notebook, Python scripts |

### 9. Dataset Details

The project uses product data from Amazon and Flipkart. The final combined dataset is stored in `combined_cleaned_data.csv`.

| Dataset File | Rows | Columns | Purpose |
|---|---:|---:|---|
| `cleaned_data.csv` | 3992 | 6 | Cleaned base product dataset |
| `amazon_flipkart_products_bigdata.csv` | 3818 | 5 | Additional product data |
| `combined_cleaned_data.csv` | 5448 | 6 | Final combined dataset used by the app |

Important columns in the final dataset:

- `Product_Name_Extracted`: Original readable product name.
- `Price`: Product price as text.
- `Type`: Product category.
- `Source`: Platform name, such as Amazon or Flipkart.
- `Product_Name_Clean`: Cleaned product name used for TF-IDF.
- `Price_Numeric`: Numeric price used for comparison.

Final dataset summary:

- Total products: **5448**
- Amazon products: **2716**
- Flipkart products: **2732**
- Price range: **Rs. 1 to Rs. 361962**
- Average price: **Rs. 26925.23**
- Major categories include laptops, shoes, smartphones, headphones, televisions, washing machines, books, refrigerators, jeans, and jackets.

### 10. Data Preprocessing

The preprocessing step is handled mainly in `create_combined.py` and `app.py`.

Preprocessing includes:

- Loading product data from CSV files.
- Standardizing column names from different datasets.
- Extracting numeric price values.
- Cleaning product names by converting text to lowercase.
- Removing special characters.
- Removing extra spaces.
- Creating a search-friendly product name column.
- Removing duplicate products based on product name and price.
- Dropping rows with missing product names or prices.

### 11. System Architecture

The system follows a simple pipeline:

1. User enters a product name in the Streamlit app.
2. The app cleans the search query.
3. The system finds the best matching product using exact match, fuzzy match, or TF-IDF match.
4. The matched product is converted into a TF-IDF vector.
5. Cosine similarity is calculated against all products.
6. Products with lower prices are filtered.
7. Similar cheaper products are sorted by similarity score.
8. Results are displayed with product name, price, source, category, similarity score, and savings.

### 12. Methodology

#### 12.1 Text Cleaning

Product names are cleaned to make matching more reliable. Special characters are removed, text is converted to lowercase, and extra spaces are normalized.

#### 12.2 Product Matching

The application uses three search strategies:

- **Exact substring match:** Fastest method; checks whether the search term appears directly in product names.
- **Fuzzy matching:** Handles spelling mistakes and partial matches using `SequenceMatcher`.
- **TF-IDF matching:** Finds the most semantically similar product name when exact and fuzzy matching do not work.

#### 12.3 TF-IDF Vectorization

TF-IDF stands for Term Frequency-Inverse Document Frequency. It converts product names into numerical vectors by giving importance to meaningful words and reducing the importance of common words.

In this project, `TfidfVectorizer` is used with:

- Maximum features: 50
- English stop words removed

#### 12.4 Cosine Similarity

Cosine similarity measures the angle between two TF-IDF vectors. A higher value means the product names are more similar.

The similarity score ranges from 0 to 1:

- 0 means not similar.
- 1 means highly similar.

#### 12.5 Price Filtering

After similarity calculation, the system filters only products where:

`Alternative Product Price < Original Product Price`

This ensures that the recommendations are actually cheaper than the searched product.

### 13. Main Modules

#### 13.1 `create_combined.py`

This script combines the cleaned base dataset with the larger Amazon-Flipkart dataset. It standardizes column names, creates cleaned product names, extracts numeric prices, removes duplicates, and saves the final dataset as `combined_cleaned_data.csv`.

#### 13.2 `app.py`

This is the main Streamlit application. It loads data, prepares the TF-IDF matrix, handles product search, finds cheaper alternatives, finds product variants, and displays the results in a web interface.

Important functions:

- `clean_text()`: Cleans product names and search queries.
- `fuzzy_match_score()`: Calculates fuzzy similarity between strings.
- `find_best_product_match()`: Finds the best product match using exact, fuzzy, and TF-IDF matching.
- `load_and_prepare_data()`: Loads the final dataset and builds the TF-IDF matrix.
- `find_cheaper_alternatives()`: Finds cheaper similar products.
- `find_same_product_variants()`: Finds similar product variants.
- `get_similar_product_suggestions()`: Suggests products when no direct match is found.

#### 13.3 Test Scripts

The project includes test scripts such as:

- `test_app_load.py`
- `test_search.py`
- `test_fix.py`
- `test_fixed_search.py`

These scripts help verify whether the app loads correctly and whether search functionality works as expected.

### 14. Application Features

- Product search by name.
- Quick example buttons.
- Cheaper alternative recommendation.
- Same-product variant search.
- Exact, fuzzy, and TF-IDF-based matching.
- Similarity score display.
- Savings calculation in rupees.
- Savings calculation in percentage.
- Sidebar database statistics.
- Debug information option.
- Interactive result table using Streamlit.
- Cached data loading for better performance.

### 15. Result and Output

The project successfully creates a working web application for finding product alternatives. When a user searches for a product such as a phone or laptop, the system identifies the best matching product and displays cheaper alternatives ranked by similarity.

The output table includes:

- Product name
- Price
- Source
- Product type
- Similarity score
- Original price
- Savings in rupees
- Savings percentage

This output helps users quickly compare products and make better purchase decisions.

### 16. Advantages

- Saves time in product comparison.
- Helps users find cheaper alternatives.
- Works across Amazon and Flipkart data.
- Handles partial and typo-based searches.
- Provides measurable similarity scores.
- Simple and user-friendly web interface.
- Uses lightweight machine learning techniques.

### 17. Limitations

- The recommendations depend on the quality and freshness of the dataset.
- Product-name similarity may not always capture full technical specification similarity.
- The system does not currently use product ratings, reviews, images, or delivery charges.
- Prices may change frequently on real e-commerce platforms.
- The dataset is static and not connected to live Amazon or Flipkart APIs.

### 18. Future Scope

- Add live web scraping or API-based price updates.
- Include product ratings and reviews in ranking.
- Compare technical specifications such as RAM, storage, processor, screen size, and battery.
- Add category-wise filtering.
- Add image-based product comparison.
- Deploy the application online.
- Add login and user watchlist features.
- Send price-drop alerts to users.

### 19. Conclusion

The Product Alternative Finder project demonstrates how text processing and machine learning can be used to solve a practical e-commerce problem. By combining TF-IDF vectorization, cosine similarity, fuzzy matching, and price filtering, the system recommends cheaper products that are similar to the user's selected product. The Streamlit interface makes the system easy to use and suitable for demonstration in a college project presentation.

### 20. Presentation Slide Outline

Use the following slide flow for your college presentation:

1. **Title Slide**
   Product Alternative Finder using TF-IDF and Cosine Similarity.

2. **Introduction**
   Explain the need for product comparison across Amazon and Flipkart.

3. **Problem Statement**
   Manual product comparison is slow and users may miss cheaper alternatives.

4. **Objectives**
   Show the main goals: clean data, search products, compare similarity, and recommend cheaper items.

5. **Dataset**
   Mention the final dataset size: 5448 products from Amazon and Flipkart.

6. **Technology Stack**
   Python, Streamlit, Pandas, Scikit-learn, TF-IDF, Cosine Similarity.

7. **System Architecture**
   Show the pipeline from user input to final recommendation.

8. **Algorithm**
   Explain exact matching, fuzzy matching, TF-IDF vectorization, cosine similarity, and price filtering.

9. **Application Features**
   Search, cheaper alternatives, same-product variants, savings display, and database stats.

10. **Demo Slide**
    Show screenshots or live demo of searching a product and viewing alternatives.

11. **Results**
    Explain how products are ranked and how savings are calculated.

12. **Limitations and Future Scope**
    Mention static dataset, changing prices, and possible live price updates.

13. **Conclusion**
    Summarize how the project helps users find similar products at lower prices.

### 21. Short Viva Questions and Answers

**Q1. What is the main purpose of this project?**  
The main purpose is to find cheaper alternatives for a searched product using product-name similarity and price comparison.

**Q2. Which algorithm is used for similarity comparison?**  
The project uses TF-IDF vectorization and cosine similarity.

**Q3. Why is TF-IDF used?**  
TF-IDF converts product names into numerical vectors and gives higher importance to meaningful words.

**Q4. What is cosine similarity?**  
Cosine similarity measures how similar two vectors are. In this project, it measures similarity between product names.

**Q5. What framework is used for the web application?**  
Streamlit is used to build the web application.

**Q6. How does the system handle spelling mistakes?**  
It uses fuzzy matching with Python's `SequenceMatcher`.

**Q7. What is the final dataset size?**  
The final combined dataset contains 5448 products.

**Q8. What are the main sources of data?**  
The data sources are Amazon and Flipkart.

**Q9. What is the biggest limitation of the project?**  
The dataset is static, so prices may not always match live e-commerce prices.

**Q10. How can the project be improved in the future?**  
It can be improved by adding live price updates, reviews, ratings, specification comparison, and deployment.

