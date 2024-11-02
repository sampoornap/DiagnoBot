# Diagnostic Medical Conversation System incorporating RAG, fine-tuned LLaMa and UMLS Knowledge Graph

This repository contains a Rasa based chat medical chat system, that can be used by anyone to gain access to medical information or analyse any health issue. The project is structured into several modules, each handling a specific aspect of the system.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Key Components](#key-components)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Extending the Project](#extending-the-project)
7. [Testing](#testing)
8. [Dependencies](#dependencies)

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
