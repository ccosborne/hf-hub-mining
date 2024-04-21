# Quantitative Analysis of Development Activity on the Hugging Face Hub

This repository contains Python scripts used for data collection and analysis in the research paper titled "The AI community building the future? A quantitative analysis of development activity on the Hugging Face Hub" by Cailean Osborne, Jennifer Ding, and Hannah Rose Kirk. The paper will be available on arXiv shortly.

## Research Questions

The paper aims to answer the following research questions:

1. **RQ1**: What are typical patterns of development activity on the HF Hub?
2. **RQ2**: What is the social network structure of the HF Hub developer community?
3. **RQ3**: What is the distribution of model adoption on the HF Hub, and who are the key actors driving the development of the most widely-adopted models?

## Data Collection Scripts

The following Python scripts are used for collecting data from the Hugging Face (HF) Hub API:

1. `hf-data-models.py`: Collects data about models on the HF Hub, including owner, model name, last modified date, license, tags, likes, downloads, commits, and discussions.
2. `hf-data-datasets.py`: Collects data about datasets on the HF Hub, including owner, model name, last modified date, license, tags, likes, downloads, commits, and discussions.
3. `hf-data-spaces.py`: Collects data about spaces on the HF Hub, including owner, space name, last modified date, license, tags, likes, commits, and discussions.
4. `hf-collabnet.py`: Collects data for building collaboration networks based on commit activity in model repositories on the HF Hub.
5. `hf-collabnet-subfields.py`: Collects data for building collaboration networks in specific subfields (Computer Vision, Natural Language Processing, and Multimodal) based on commit activity and repository tags.
6. `hf-spaces-dependencies.py`: Collects data on model dependencies in spaces on the HF Hub.
7. `hf-spaces-dependencies-subfields.py`: Collects data on model dependencies in spaces for specific subfields (Computer Vision, Natural Language Processing, and Multimodal) based on repository tags.

We have not included our `config.py` file but remember use one and to store your token there.

## Requirements

To run the scripts, you need Python 3.x and the following packages:
- pandas
- huggingface_hub
- numpy
