# 🛍️ Product Alternative Finder - Web App

A Streamlit web application that finds cheaper products similar to your favorite items using TF-IDF and Cosine Similarity.

##  Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Features

**Smart Search**
- Search for any product by name
- Case-insensitive matching
- Quick example buttons for popular products

**Price Comparison**
- Shows original product price and alternatives
- Displays savings in rupees and percentage
- Ranked by similarity and price

**Similarity Scoring**
- Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- Cosine similarity scoring (0-1 scale, higher = more similar)
- Only recommends CHEAPER alternatives

**Performance**
- Data is cached for fast loading
- Lightning-fast similarity searches
- Handles 176+ products instantly

##  How It Works

1. **Input**: User searches for a product
2. **Search**: Finds matching product in database
3. **Vectorize**: Converts product name to numerical vector
4. **Compare**: Calculates similarity with all products
5. **Filter**: Keeps only cheaper alternatives
6. **Rank**: Sorts by similarity score (highest first)
7. **Output**: Displays results with price savings

##  Example Usage

1. Search for "HP Victus"
2. Get cheaper laptop alternatives like "HP 15" with 30%+ savings
3. See similarity score and exact price difference

##  File Structure

```
.
├── app.py                 # Streamlit web app
├── eda.ipynb             # Jupyter notebook with model development
├── cleaned_data.csv      # Product database (176 products)
└── requirements.txt      # Python dependencies
```

##  Technical Details

- **Framework**: Streamlit
- **ML Algorithm**: TF-IDF Vectorizer + Cosine Similarity
- **Data**: 176 products from Amazon & Flipkart
- **Features**: Product names, prices, sources, types

##  Database Info

- **Total Products**: 176
- **Sources**: Amazon, Flipkart
- **Categories**: Mobiles, Laptops
- **Price Range**: ₹7,999 - ₹1,29,999

##  Tips

- Use product brand names for best results (e.g., "Samsung", "iPhone", "Dell")
- Adjust the "Number of alternatives" slider to see more or fewer results
- Check the sidebar for database statistics

##  Customization

To modify the app:
1. Edit `app.py` for UI changes
2. Modify `find_cheaper_alternatives()` function for algorithm changes
3. Update `requirements.txt` if adding new dependencies

##  Notes

- Ensure `cleaned_data.csv` is in the same directory as `app.py`
- The model is cached automatically for performance
- Works on Windows, Mac, and Linux

---

Enjoy finding great deals! 
