# COMP3004 Designing Intelligent Agents
***Interactive Story Generation Agent***

## Introduction
This project developed an autonomous intelligent agent based on language, which can generate short fictional stories according to users' instructions.
The system compared various language modeling methods for generating story plots based on prompts, including:

- Bigram
- Trigram
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory Networks (LSTM)
- Transformer-based LLM (TinyLlama)

The project also included an automated evaluation framework based on indicators, including:

- Keyword Coverage
- Semantic Coherence
- Grammar Validity
- Vocabulary Diversity
- Repetition Rate

The experimental interface and story generation environment were implemented using Streamlit.

## Running
*Note:* When Streamlit is executed for the first time on a new machine, users can simple press "Enter" to skip the optinal email sign in and continue running the application.
### Configuration (Install)
1. Final Install
```bash
pip install -r requirements.txt
pip install torchvision
pip install -U sentence-transformers
pip install spacy
python -m spacy download en_core_web_sm
# Or (if fail) in the lab -- to use the related python environment
C:\Anaconda3\python.exe -m spacy download en_core_web_sm
# Run:
python -m streamlit run app.py
# Or (if fail) in the lab -- to use the related python environment
C:\Anaconda3\python.exe -m streamlit run app.py
```

2. Installation during Development
```bash
pip install streamlit
pip install datasets
pip install transformers

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

pip install torch torchvision torchaudio

pip install spacy

pip install langchain-community langchain-core
pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3

pip install -U sentence-transformers

```

### Dataset
This project uses the TinyStories dataset:

- https://huggingface.co/datasets/roneneldan/TinyStories


## Model Files
Statistical models and recurrent models, as lightweight pre-training model files, are included in the following content: `saved_models/`

Due to the size limit of the submitted course assignment files, the large cache of Transformer model weights files and the complete TinyStories dataset were not included in this submission.

If the network connection is available, the Transformer model will automatically download and cache locally during its first run.


## Overall Project Files and Reproducibility
1. The **ZIP file** and **GitHub repository** contain the simplified version of this project.

2. Due to the size limit for course assignment submission files, the following larger files are not included in the submitted compressed file:

   - Text files for completing the TinyStories dataset
   - Cached checkpoint files and tokenizer files for the Transformer model
   - Large local cache files of HuggingFace models
   The complete and full version of the project, including all datasets and local cached large language model files, is provided separately via ***OneDrive***:
   - Overall project files (***OneDrive***):
    *https://uniofnottm-my.sharepoint.com/:f:/g/personal/scyxy5_nottingham_ac_uk/IgBKnw-EzoVFSqd9F059rfRPAToHAaH4jrt_PGg1SUeNcVY?e=gd8Vk4*
    
   - A lightweight project repository can also be obtained on GitHub (***GitHub repository:***):
    *https://github.com/Y7-hista/COMP3004-Story-Agent-Project.git*

3. The ZIP file submission package and GitHub repository contain all source code, evaluation system, Streamlit interface, saved statistical models, and recurrent neural network models, which are necessary for reproducing the main experimental framework.

If a network connection is available, the Transformer model will be automatically downloaded and cached locally by HuggingFace Transformers on the first run.

This separation is necessary because the complete dataset and Transformer cache files exceed the size limit for Moodle course assignment uploads.