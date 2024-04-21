import json
import pandas as pd
from huggingface_hub import HfApi, get_repo_discussions
import numpy as np
import os 
from requests.exceptions import HTTPError
import time
from config import token

# Check if the Data folder exists, if not create it
if not os.path.exists("/home/user/HF-Analysis/Data"):
    os.makedirs("/home/user/HF-Analysis/Data")

# Call API
print('1/3: Calling API for models on HF Hub')
api = HfApi(token=token)
model_list = list(api.list_models())
model_count = len(model_list)

# Wrangle data into a pandas dataframe
print(f'2/3: Wrangling data for {model_count} models')
model_data_rows = []
for model in model_list:
    # Get the owner and model name from the 'modelId' field
    if '/' in model.modelId:
        owner, model_name = model.modelId.split('/', 1)
    else:
        owner = np.nan
        model_name = model.modelId  # assign 'modelId' to 'model_name' if no '/' is found

    # Get license info from tags
    license = next((tag.split(":")[1] for tag in model.tags if "license:" in tag), np.nan)

    # Use try-except block for retrieving commits data since some repositories are private 
    try:
        # Get all commits in a list
        commits = list(api.list_repo_commits(repo_id=model.modelId))
        # Count commits
        num_commits = len(commits)
        # Count commit contributors
        commit_contributors = set(author for commit in commits for author in commit.authors)
    except Exception as e:  # Catch any error that may occur during the commits operations
        print(f"Error retrieving commits data for model {model.modelId}: {str(e)}")
        num_commits = np.nan
        commit_contributors = set()

    # Use try-except block for retrieving discussions data since some repositories are private 
    try:
        # Get all discussions in a list
        discussions = list(get_repo_discussions(repo_id=model.modelId)) 
        # Count discussions
        num_discussions = len(discussions)
        # Count discussions contributors
        discussions_contributors = set(discussion.author for discussion in discussions)
    except Exception as e:  # Catch any error that may occur during the discussions operations
        print(f"Error retrieving discussions data for model {model.modelId}: {str(e)}")
        num_discussions = np.nan
        discussions_contributors = set()

    # Prepare a row to add to the list
    row_to_add = pd.Series({
        'owner': owner,
        'model': model_name,
        'id': model.modelId,
        'lastModified': model.lastModified if model.lastModified else np.nan,
        'license': license,
        'tags': ', '.join(model.tags if model.tags else []),
        'likes': model.likes if model.likes else np.nan,
        'downloads': model.downloads if model.downloads else np.nan,
        'commits': num_commits,
        'commits_contributors': commit_contributors,
        'discussions': num_discussions,
        'discussions_contributors': discussions_contributors
    })
    
    # Append the row to the list
    model_data_rows.append(row_to_add)

    # Wait 1 second before wrangling data of the next model
    time.sleep(1)

# Convert the list to a DataFrame
df = pd.DataFrame(model_data_rows)

# Convert non-null values to int64
for column in ['downloads', 'likes', 'commits', 'discussions']:
    df.loc[df[column].notna(), column] = df.loc[df[column].notna(), column].astype(np.int64)

# Convert 'lastModified' to YYYY-MM-DD format
df.loc[df['lastModified'].notna(), 'lastModified'] = pd.to_datetime(df.loc[df['lastModified'].notna(), 'lastModified']).dt.strftime('%Y-%m-%d')

# Save the DataFrame to a CSV file
try:
    df.to_csv('/home/user/HF-Analysis/Data/hf-data-models.csv', index=False)
    print(f'3/3: Saved data for {model_count} models')
except IOError as e:
    print(f"Error saving data to CSV: {str(e)}")