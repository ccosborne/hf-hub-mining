import json
import pandas as pd
from huggingface_hub import HfApi, get_repo_discussions
from huggingface_hub.utils._validators import HFValidationError
import numpy as np
import os 
from requests.exceptions import HTTPError
from config import token
import time

# Check if the Data folder exists, if not create it
if not os.path.exists("/home/user/HF-Analysis/Data"):
    os.makedirs("/home/user/HF-Analysis/Data")

# Call API
print('1/3: Calling API for spaces on HF Hub')
api = HfApi(token=token)
space_list = list(api.list_spaces())
space_count = len(space_list)

# Wrangle data into a pandas dataframe
print(f'2/3: Wrangling data for {space_count} spaces')
space_data_rows = []
for space in space_list:
    # Get the organisation and space name from the 'id' field
    if '/' in space.id:
        organisation, space_name = space.id.split('/', 1)
    else:
        organisation = np.nan
        space_name = space.id  # assign 'id' to 'space_name' if no '/' is found

    # Get license info from tags
    license = next((tag.split(":")[1] for tag in space.tags if "license:" in tag), np.nan)

    # Use try-except block for retrieving commit data since some repositories are private 
    try:
        # Get all commits in a list
        commits = list(api.list_repo_commits(repo_id=space.id, repo_type="space"))
        # Count commits
        num_commits = len(commits)
        # Count commit contributors
        commit_contributors = set(author for commit in commits for author in commit.authors)
    except Exception as e:  # Catch any error that may occur during the commits operations
        print(f"Error retrieving commits data for space {space.id}: {str(e)}")
        num_commits = np.nan
        commit_contributors = set()

    # Use try-except block for retrieving discussions data since some repositories are private 
    try:
        # Get all discussions in a list
        discussions = list(get_repo_discussions(repo_id=space.id, repo_type="space")) 
        # Count discussions
        num_discussions = len(discussions)
        # Count discussions contributors
        discussions_contributors = set(discussion.author for discussion in discussions)
    except Exception as e:  # Catch any error that may occur during the discussions operations
        print(f"Error retrieving discussions data for space {space.id}: {str(e)}")
        num_discussions = np.nan
        discussions_contributors = set()

    # Prepare a row to add to the list
    row_to_add = pd.Series({
        'owner': organisation,
        'space': space_name,
        'id': space.id,
        'lastModified': space.lastModified if space.lastModified else np.nan,
        'license': license, 
        'tags': ', '.join(space.tags if space.tags else []),
        'likes': space.likes if space.likes else np.nan,
        'commits': num_commits,
        'commits_contributors': commit_contributors,
        'discussions': num_discussions,
        'discussions_contributors': discussions_contributors
    })
    
    # Append the row to the list
    space_data_rows.append(row_to_add)

    # Wait 1 second before wrangling data of the next space
    time.sleep(1)

# Convert the list to a DataFrame
df = pd.DataFrame(space_data_rows)

# Convert non-null values to int64
for column in ['likes', 'commits', 'discussions']:  
    if column in df.columns:  
        df.loc[df[column].notna(), column] = df.loc[df[column].notna(), column].astype(np.int64)

# Convert 'lastModified' to YYYY-MM-DD format
if 'lastModified' in df.columns:
    df.loc[df['lastModified'].notna(), 'lastModified'] = pd.to_datetime(df.loc[df['lastModified'].notna(), 'lastModified']).dt.strftime('%Y-%m-%d')

# Save the DataFrame to a CSV file
try:
    df.to_csv('/home/user/HF-Analysis/Data/hf-data-spaces.csv', index=False)
    print(f'3/3: Saved data for {space_count} spaces')
except IOError as e:
    print(f"Error saving data to CSV: {str(e)}")