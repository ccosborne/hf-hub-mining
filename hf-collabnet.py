import pandas as pd
from huggingface_hub import HfApi
import os
from config import token

# Check if the Data folder exists, if not create it
data_folder = '/home/user/HF-Analysis/Data'
os.makedirs(data_folder, exist_ok=True)

# Call API to get model IDs
print('1/3: Retrieving models from HF model hub')
api = HfApi(token=token)
model_ids = [model.modelId for model in api.list_models()]

# Initialize counters and lists
edgelist = []
models_count = len(model_ids)
public_repositories = 0
repositories_with_commits = 0
repositories_only_one_contributor = 0

# Get commit data for each model (skip private repositories)
print(f'2/3: Retrieving commit data for {models_count} models')
for model_id in model_ids:
    try:
        public_repositories += 1
        commits = api.list_repo_commits(repo_id=model_id, token=token)

    except Exception as e:  # Skip private repositories
        print(f'Error with: {model_id}: {str(e)}')  # Print error for troubleshooting
        continue

    unique_authors = list(set(author for commit in commits for author in commit.authors))

    for commit in commits:
        source_author = commit.authors[0]
        for target_author in unique_authors:
            if source_author != target_author:
                edgelist.append((source_author, target_author))

        if len(unique_authors) == 1:
            edgelist.append((source_author, source_author))  # Create self-loop edges for single-contributor repositories

    repositories_with_commits += len(commits) > 0
    repositories_only_one_contributor += len(unique_authors) == 1

# Save data
edgelist_df = pd.DataFrame(edgelist, columns=['source', 'target'])
edgelist_df = edgelist_df.groupby(['source', 'target']).size().reset_index(name='freq')
edgelist_df.to_csv(os.path.join(data_folder, 'hf-edgelist-contributors.csv'), index=False)
print(f'3/3: Saved commit data from {repositories_with_commits} model repositories.')
print(f'\nKey facts:\n - Model repositories: {public_repositories}')
print(f' - Model repositories with commits: {repositories_with_commits} ({repositories_with_commits / public_repositories:.2%})')
print(f' - Model repositories with zero commits: {public_repositories - repositories_with_commits} ({(public_repositories - repositories_with_commits) / public_repositories:.2%})')
print(f' - Model repositories with only one contributor: {repositories_only_one_contributor} ({repositories_only_one_contributor / repositories_with_commits:.2%})')