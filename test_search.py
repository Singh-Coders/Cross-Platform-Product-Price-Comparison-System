import pandas as pd

df = pd.read_csv('combined_cleaned_data.csv')

# Test the exact search
search_term = 'Samsung 55 4K TV'
matching = df[df['Product_Name_Extracted'].str.contains(search_term, case=False, na=False)]
print(f'Search term: "{search_term}"')
print(f'Matches found: {len(matching)}\n')

# Try with partial match
search_term2 = 'Samsung 55'
matching2 = df[df['Product_Name_Extracted'].str.contains(search_term2, case=False, na=False)]
print(f'Partial search: "{search_term2}"')
print(f'Matches found: {len(matching2)}\n')

print('Sample TV products in database:')
tvs = df[df['Type'] == 'Television']
print(tvs['Product_Name_Extracted'].head(10).to_string())
print(f'\nTotal TVs: {len(tvs)}')

# Check exact names
print(f'\n\nExact unique Samsung 55 TV names:')
samsung_55_tvs = df[df['Product_Name_Extracted'].str.contains('Samsung 55', case=False, na=False)]
print(samsung_55_tvs['Product_Name_Extracted'].unique())
