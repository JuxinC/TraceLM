# About
This is a research-oriented framework designed to investigate the effectiveness of large language models (LLMs) in automating the trace link recovery of issue–commit in software repositories.
## Dataset

This project uses the **SEOSS 33** dataset, which contains a large number of typed artifacts and trace links between them..

- Dataset link: [https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PDDZ4Q](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PDDZ4Q)

If you use this dataset, please cite the following paper:

> Rath, M., & Mäder, P. (2019). The SEOSS 33 dataset—Requirements, bug reports, code history, and trace links for entire projects. *Data in Brief*, 25, 104005.  
> [https://doi.org/10.1016/j.dib.2019.104005](https://doi.org/10.1016/j.dib.2019.104005)

## Key Features
- Full pipeline：From raw SEOSS 33 data to final prediction output, covering preprocessing, feature extraction, and evaluation.
- Textual similarity features (Semantic similarity models supported):
  - VSM + TF-IDF 
  - Word2Vec
  - FastText
  - SentenceTransformer (all-MiniLM-L6-v2)
  - OpenAI Embeddings
- Process-related features
  - Temporal (e.g., issue–commit temporal information comparison)
  - Author matching (linking based on commit author and issue assignee/ reporter)
- Multiple evaluation modes
  - Threshold-based linking (e.g., based on cosine similarity)
  - Classifier-based prediction using: Random Forest, XGBoost
- Evaluation metrics:
  - the feature importance of the fitted models
  - precision, recall, f2-score, f0.5-score of the fitted models  

## Project Structure
```text
TraceLM/
├── notebooks/
│   └── TraceLM_main.ipynb            # Main Jupyter Notebook: full pipeline for data loading → cleaning → feature engineering → model evaluation
│
├── src/                              # Source code (modular functions used in the notebook)
│   ├── load_data/
│   │   └── data_loader.py            # Load raw JIRA & SVN data from SQLite, remove illegal chars
│
│   ├── clean_data/
│   │   ├── cleanCommitData.py        # Clean SVN commit logs
│   │   ├── cleanJiraData.py          # Clean JIRA issue texts
│   │   ├── subsetAccordTime.py       # Subset data by time window
│   │   ├── checkValidityTrace.py     # Label trace validity based on known links
│   │   └── createCorpusFromDocumentList.py  # Tokenization + cleaning for corpora
│
│   ├── features_engineering/
│   │   ├── calculateTimeDifference.py   # Compute time deltas (commit - issue)
│   │   └── checkAuthorMatch.py         # Whether SVN committer matches JIRA assignee/reporter
│
│   ├── model_similarity/
│   │   ├── embedding_choice.py         # Utility for selecting embedding models
│   │   ├── createFittedTF_IDF.py       # TF-IDF vectorizer fitting + cosine similarity
│   │   ├── wordToVec.py                # Word2Vec cosine similarity
│   │   ├── fastText.py                 # FastText cosine similarity
│   │   ├── sentenceTransformer.py      # SBERT-based similarity + embedding processing
│   │   └── openAI.py                   # OpenAI embeddings (text-embedding-ada-002)
│
├── data/
│   ├── raw_data/                     # SEOSS33 SQLite3 input files (external, not included in repo)
│   ├── intermediate/                 # Cached cleaned data, corpora, cartesian products
│   ├── features/                     # Engineered features: process-based, IR, Word2Vec, etc.
│   ├── models/                       # Trained embedding models (Word2Vec, FastText)
│   └── results/                      # Evaluation results (classification metrics, plots, Excel)
│
├── requirements.txt                  # List of dependencies
└── README.md                         # Project description (this file)
```
##  Trained embedding models (Word2Vec, FastText)
You can directly download our pre-trained Word2Vec and FastText models, which were trained on data from 8 projects, from the links below. After downloading, please place them into the data/model folder
https://drive.google.com/drive/folders/1Gi3esNwWa8YsPzfsWaSuarmEWwquWsme?usp=drive_link
