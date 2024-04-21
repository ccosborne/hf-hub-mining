import pandas as pd
from huggingface_hub import HfApi
import os
import time
from config import token

def collect_data(tag_list, output_file):
    print(f'1/4: Retrieving models from HF model hub for tags: {tag_list}')
    api = HfApi(token=token)
    models_list = api.list_models()
    model_ids = [model.modelId for model in models_list if any(tag in model.tags for tag in tag_list)]

    print(f'2/4: Scraping data from public repositories')
    models = []
    for model_id in model_ids:
        try:
            model_info = api.model_info(repo_id=model_id)
            models.append(model_info)
            time.sleep(0.5)  # sleep for half a second to avoid rate limiting
        except Exception as e: # Skip private repositories
            print(f'Error with model {model_id}: {str(e)}')  # Print error for troubleshooting
            continue

    print(f'3/4: Creating dataframe for space dependencies')
    edges_spaces = [(space, model.modelId) for model in models for space in model.spaces]
    edgelist_spaces = pd.DataFrame(edges_spaces, columns=['source', 'target'])

    print(f'4/4: Saved CSV of space dependencies for tags: {tag_list}')
    edgelist_spaces.to_csv(output_file, index=False)

# Define your tag lists and output file names
computervision_tags = ['computer-vision', 'depth-estimation', 'image-classification', 'object-detection', 'image-segmentation', 'image-to-image', 'unconditional-image-generation', 'video-classification', 'zero-shot-image-classification']
nlp_tags = ['text-classification', 'token-classification', 'table-question-answering', 'question-answering', 'zero-shot-classification', 'translation', 'summarization', 'conversational', 'text-generation', 'text2text-generation', 'fill-mask', 'sentence-similarity']
multimodal_tags = ['multimodal', 'feature-extraction', 'text-to-image', 'image-to-text', 'text-to-video', 'visual-question-answering', 'document-question-answering', 'graph-machine-learning']

# Specify output file names for each tag list
output_file_computervision = '/home/user/HF-Analysis/Data/hf-edgelist-spaces-dependencies-cv.csv'
output_file_nlp = '/home/user/HF-Analysis/Data/hf-edgelist-spaces-dependencies-nlp.csv'
output_file_multimodal = '/home/user/HF-Analysis/Data/hf-edgelist-spaces-dependencies-mm.csv'

# Collect data for each tag list
collect_data(computervision_tags, output_file_computervision)
collect_data(nlp_tags, output_file_nlp)
collect_data(multimodal_tags, output_file_multimodal)