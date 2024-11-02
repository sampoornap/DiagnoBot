# Diagnostic Medical Conversation System incorporating RAG, fine-tuned LLaMa and UMLS Knowledge Graph

This repository contains a Rasa based chat medical chat system, that can be used by anyone to gain access to medical information or analyse any health issue. The project is structured into several modules, each handling a specific aspect of the system.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Key Components](#key-components)
4. [Dependencies](#dependencies)

## Project Overview

This project implements a robust pipeline incorporating Rasa, LangChain, various LLMs, veector databases and knowledge graphs.

## Directory Structure

```
├── README.md
├── actions/
│   ├── langchain_app/            # contains docker config for the RAG mechanism
│   ├── actions.py                # Rasa custom actions
│   └── data.csv                  # Medical conditions dataset
├── custom_components/            # spell checker + NER  
├── data/                   
│   ├── nlu.yml                   # Rasa NLU
│   ├── rules.yml                 # conversation rules
│   └── stories.yml               # training stories
├── docs/                         # frontend stuff
├── exp/                          # RAG experiments
├── extras/                       # legacy modules
├── models/                       # trained models
├── results/                      # Rasa visualizations
├── static/                       # frontend
├── tests/                        # test stories
├── train_test_split/             # train/test nlu
├── config.yml                    # Rasa config
├── credentials.yml               # credentials for the voice & chat platforms
├── domain.yml                    # all Rasa stuff
├── endpoints.yml                 # different endpoints for bot
├── index.html                    # frontend
├── requirements.txt              
├── story_graph.dot               
└── test_medllama.py              # finetuned medllama model
```


## Key Components

### 1. Actions Module (`actions/`)

- Contains custom actions that are triggered based on user message such as asking medical diagnosis questions, answering follow-up questions and generating doctor report.
- Contains a langchain based mechanism that has embeddings of wikipedia summaries of all diseases listed by the NIH stored in a pinecone database.
- Patient information is converted to embeddings and a similarity search algorithm ranks the top n possible diagnoses/medications. Response generation is done using fine-tuned LLama or mistral.

### 2. Data Module (`data/`)

- Contains configuration files for the Rasa chat system.

### 3. Custom Components Module (`custom_components/`)

- Has a custom spell checker and medical NER tagging mechanism.


## Dependencies

- Rasa
- PyTorch
- LangChain
- Cohere
- Wikipedia-API
- Tensorflow
- Tokenizers
- GitPython

For a complete list of dependencies, refer to `requirements.txt`.

---
