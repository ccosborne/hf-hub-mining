import pandas as pd
from fuzzywuzzy import fuzz
import string
from multiprocessing import Pool

# File paths
realnames_file = '/home/user/HF-Analysis/Data/realnames.csv'
usernames_file = '/home/user/HF-Analysis/Data/usernames.csv'
output_file = '/home/user/HF-Analysis/Data/realname-matches-90.csv'

# Constants
THRESHOLD = 90

# Function to preprocess usernames
def preprocess_username(username):
    username = username.lower()
    username = ''.join(char for char in username if char not in string.punctuation)
    username = username.replace(" ", "")
    return username

print('1/4: Importing and preprocessing data')
realnames_data = pd.read_csv(realnames_file, dtype=str)
usernames_data = pd.read_csv(usernames_file, dtype=str)

realnames_data['realname'] = realnames_data['realname'].astype(str)
usernames_data['username'] = usernames_data['username'].astype(str)

# Create sets of preprocessed usernames for faster membership tests
usernames_set = set(map(preprocess_username, usernames_data['username']))

print('2/4: Finding potential matches')

# Function to find potential matches for a realname
def find_matches(realname):
    original_realname = realname
    preprocessed_realname = preprocess_username(original_realname)
    
    matches = []
    for preprocessed_username in usernames_set:
        similarity = fuzz.token_set_ratio(preprocessed_realname, preprocessed_username)
        if similarity >= THRESHOLD:
            original_username = usernames_data[usernames_data['username'].apply(preprocess_username) == preprocessed_username]['username'].values[0]
            matches.append((original_realname, original_username, similarity))
    
    return matches

# Use multiprocessing to parallelize the process
pool = Pool(processes=4)  
potential_matches = pool.map(find_matches, realnames_data['realname'])
pool.close()
pool.join()

# Flatten the list of potential matches
potential_matches = [match for matches in potential_matches for match in matches]

# Create a DataFrame from the potential matches
potential_matches_df = pd.DataFrame(potential_matches, columns=['realname', 'username', 'similarity'])

# Drop the matches where the values are identical
potential_matches_df = potential_matches_df[potential_matches_df['realname'] != potential_matches_df['username']]

# Drop duplicates based on the sorted combinations, keeping the first occurrence
potential_matches_df['Sorted_Username'] = potential_matches_df.apply(lambda row: tuple(sorted([row['realname'], row['username']])), axis=1)
potential_matches_df.drop_duplicates(subset='Sorted_Username', keep='first', inplace=True)
potential_matches_df.drop(columns='Sorted_Username', inplace=True)

print('3/4: Found', len(potential_matches_df), 'potential matches.')

# Save the potential_matches_df DataFrame to a CSV file
potential_matches_df.to_csv(output_file, index=False)
print('4/4: Results saved to', output_file)
