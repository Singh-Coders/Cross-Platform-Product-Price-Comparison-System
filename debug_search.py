import pandas as pd
import re

df = pd.read_csv('combined_cleaned_data.csv')

# Test different search approaches
search_term = 'Samsung 55 4K TV'
clean_search = re.sub(r'[^a-zA-Z0-9\s]', '', search_term)

print(f"Original search: '{search_term}'")
print(f"Cleaned search: '{clean_search}'")
print()

# Check what's in the database
print("Exact names in DB:")
samsung_tv_matches = df[df['Product_Name_Extracted'].str.contains('Samsung 55', case=False, na=False)]
print(samsung_tv_matches['Product_Name_Extracted'].head(5).to_list())
print()

# Create a cleaned version of database for matching
df['Product_Name_Cleaned_Search'] = df['Product_Name_Extracted'].apply(
    lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)).lower()
)

# Now search in cleaned column
matches_method1 = df[df['Product_Name_Extracted'].str.contains(search_term, case=False, na=False, regex=False)]
print(f"Method 1 (original search): {len(matches_method1)} matches")

matches_method2 = df[df['Product_Name_Extracted'].str.contains(clean_search, case=False, na=False, regex=False)]
print(f"Method 2 (cleaned search in original): {len(matches_method2)} matches")

matches_method3 = df[df['Product_Name_Cleaned_Search'].str.contains(clean_search.lower(), case=False, na=False, regex=False)]
print(f"Method 3 (cleaned search in cleaned): {len(matches_method3)} matches")

if len(matches_method3) > 0:
    print(f"\n✅ FOUND using Method 3!")
    print(f"First match: {matches_method3.iloc[0]['Product_Name_Extracted']}")
    print(f"Price: ₹{matches_method3.iloc[0]['Price_Numeric']}")
