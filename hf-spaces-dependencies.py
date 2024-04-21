import pandas as pd
from huggingface_hub import HfApi
import os
import time
from config import token

# Check if the Data folder exists, if not create it
data_folder = '/home/user/HF-Analysis/Data'
os.makedirs(data_folder, exist_ok=True)

# Call API
print('1/4: Retrieving models from HF model hub.')
api = HfApi(token=token)
models_list = api.list_models()
model_ids = [model.modelId for model in models_list]

print(f'2/4: Fetching info for public repositories')
models = []
for model_id in model_ids:
    try:
        model_info = api.model_info(repo_id=model_id)
        models.append(model_info)
        time.sleep(0.5)  # sleep for half a second to avoid rate limiting
    except Exception as e: # Skip private repositories
        print(f'Error with model {model_id}: {str(e)}')  # Print error for troubleshooting
        continue

print('3/4: Creating dataframe for space dependencies...')
edges_spaces = [(space, model.modelId) for model in models for space in model.spaces]
edgelist_spaces = pd.DataFrame(edges_spaces, columns=['source', 'target'])

edgelist_spaces.to_csv(os.path.join(data_folder, 'hf-edgelist-spaces-dependencies.csv'), index=False)
print('4/4: Saved CSV of space dependencies')