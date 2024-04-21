import pandas as pd
from fuzzywuzzy import fuzz
import string

# File paths
edgelist_file = '/home/user/HF-Analysis/Data/precleaning/hf-edgelist-contributors-similarity.csv'
output_file = '/home/user/HF-Analysis/Data/username-matches-90.csv'

# Constants
THRESHOLD = 90

# Function to preprocess usernames
def preprocess_username(username):
    username = username.lower()
    username = ''.join(char for char in username if char not in string.punctuation)
    username = username.replace(" ", "")
    return username

# Load edgelist data
df = pd.read_csv(edgelist_file, dtype={'source': str, 'target': str, 'freq': int})
df['source'] = df['source'].astype(str)
df['target'] = df['target'].astype(str)
print('1/4: Imported data')

# Remove self-loops
df = df[df['source'] != df['target']]

# Create a dictionary to map preprocessed usernames to their original values
username_to_original = {}

print('2/4: Finding potential matches')
for index, row in df.iterrows():
    original_source = row['source']
    original_target = row['target']
    
    preprocessed_source = preprocess_username(original_source)
    preprocessed_target = preprocess_username(original_target)
    
    username_to_original[preprocessed_source] = original_source
    username_to_original[preprocessed_target] = original_target

# Create a DataFrame to store the potential matches
potential_matches_df = pd.DataFrame(columns=['Username-1', 'Username-2', 'Similarity'])

# Iterate through the usernames and run similarity analysis
for preprocessed_source, original_source in username_to_original.items():
    for preprocessed_target, original_target in username_to_original.items():
        similarity = fuzz.token_set_ratio(preprocessed_source, preprocessed_target)
        if similarity >= THRESHOLD:
            potential_matches_df = potential_matches_df.append({'Username-1': original_source, 'Username-2': original_target, 'Similarity': similarity}, ignore_index=True)

# Drop the matches where the values are identical
potential_matches_df = potential_matches_df[potential_matches_df['Username-1'] != potential_matches_df['Username-2']]

# Drop duplicates based on the sorted combinations, keeping the first occurrence
potential_matches_df['Sorted_Username'] = potential_matches_df.apply(lambda row: tuple(sorted([row['Username-1'], row['Username-2']])), axis=1)
potential_matches_df.drop_duplicates(subset='Sorted_Username', keep='first', inplace=True)
potential_matches_df.drop(columns='Sorted_Username', inplace=True)
potential_matches_df.reset_index(drop=True, inplace=True)

# Print match count
print('3/4: Found', len(potential_matches_df), 'potential matches.')

# Save the potential_matches_df DataFrame to a CSV file
potential_matches_df.to_csv(output_file, index=False)
print('4/4: Results saved to', output_file)