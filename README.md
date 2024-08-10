# llm-knowledge-function
大模型知识库工具, 实现本地知识库入向量库，根据知识查询

## 安装
```bash
pip install git+https://github.com/aigc-open/llm-knowledge-function.git
```

## 使用方法
```python
from llm_knowledge_function import LocalKnowledge
from fire import Fire


def split_documents_MD():
    res = LocalKnowledge().split_documents(filename="README.MD")
    print(res)


def split_documents_PDF():
    res = LocalKnowledge().split_documents(filename="test.pdf")
    print(res)


def split_documents_PY():
    res = LocalKnowledge().split_documents(filename="test.py", chunk_size=100)
    print(res)


def filename_to_milvus():
    # res = LocalKnowledge(uri="./.demo-milvus.db",
    #                      model_name="moka-ai/m3e-base",
    #                      cache_folder="/root/.cache/huggingface/hub/").filename_to_milvus(filename="test.pdf", chunk_size=100, namespace="pdf")
    res = LocalKnowledge(uri="./.demo-milvus.db",
                         model_name="moka-ai/m3e-base",
                         cache_folder="/root/.cache/huggingface/hub/").filename_to_milvus(filename="test.py", chunk_size=100, namespace="3")
    print(res)


def knowledge_search():
    k = LocalKnowledge(uri="./.demo-milvus.db",
                       model_name="moka-ai/m3e-base",
                       cache_folder="/root/.cache/huggingface/hub/")
    res = k.similarity_search(query="import", expr='namespace == "3"')
    print(res)
    # res = k.similarity_search(query="web3.0是什么", expr='namespace == "pdf"')
    # print(res)
    # res = k.similarity_search(query="web3.0是什么", expr='namespace == "py"')
    # print(res)
    # res = k.similarity_search(query="投资的意义", expr='namespace == "pdf"')
    # print(res)


if __name__ == "__main__":
    Fire()

```
