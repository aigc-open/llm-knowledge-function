
from unstructured.nlp.tokenize import download_nltk_packages
from llm_knowledge_function import LocalKnowledge

if __name__ == "__main__":
    download_nltk_packages()
    LocalKnowledge(uri="./.demo-milvus.db",
                       model_name="moka-ai/m3e-base",
                       cache_folder="/root/.cache/huggingface/hub/",
                       local_files_only=False)
