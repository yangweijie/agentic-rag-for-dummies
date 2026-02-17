<p align="center">
<img alt="Agentic RAG for Dummies Logo" src="assets/logo.png" width="350px">
</p>

<h1 align="center">Agentic RAG 入门指南</h1>

<p align="center">
  <strong>使用 LangGraph、对话记忆和人在环中的查询澄清功能构建生产级 Agentic RAG 系统</strong>
</p>

<p align="center">
  <a href="#概述">概述</a> •
  <a href="#工作原理">工作原理</a> •
  <a href="#llm-提供商配置">LLM 提供商</a> •
  <a href="#实现">实现</a> •
  <a href="#安装与使用">安装与使用</a> •
  <a href="#故障排除">故障排除</a>
</p>

<p align="center">
  <strong>快速开始 👉</strong> 
  <a href="https://colab.research.google.com/gist/GiovanniPasq/ddfc4a09d16b5b97c5c532b5c49f7789/agentic_rag_for_dummies.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"/>
  </a>
</p>

<p align="center">
  <img alt="Agentic RAG Demo" src="assets/demo.gif" width="650px">
</p>

<p align="center">
  <strong>如果喜欢这个项目，给个 star ⭐️ 吧 :)</strong>
</p>

## 概述

本仓库演示了如何使用 LangGraph 以最少的代码构建一个 **Agentic RAG（检索增强生成）** 系统。它实现了：

- 💬 **对话记忆**：在多个问题中保持上下文，实现自然对话
- 🔄 **查询澄清**：自动重写模糊的查询或请求澄清
- 🔍 **分层索引**：搜索小而具体的块（子块）以获得精确性，检索更大的父块以获取上下文
- 🤖 **智能体编排**：使用 LangGraph 协调整个工作流程
- 🧠 **智能评估**：在细粒度块级别评估相关性
- ✅ **自我纠正**：如果初始结果不足，重新查询
- 🔀 **多智能体 Map-Reduce**：将查询分解为并行子查询，以获得全面的答案

---

### 🎯 使用本仓库的两种方式

**1️⃣ 学习路径：交互式笔记本**  
逐步教程，非常适合理解核心概念。如果您是 Agentic RAG 的新手或想快速实验，从这里开始。专注于基本工作流程，不使用高级功能，以保持简单。

**2️⃣ 构建路径：模块化项目**  
模块化架构，每个组件都可以独立替换。如果您想构建实际应用或根据需要自定义系统，请使用此方法。

**您可以自定义的内容示例：**
- **LLM 提供商**：从 Ollama 一行代码切换到 Claude、OpenAI 或 Gemini
- **智能体工作流程**：在图中添加/删除节点，并为特定领域（法律、医疗等）自定义系统提示
- **PDF 转换**：用 Docling、PaddleOCR 或其他工具替换 PyMuPDF
- **嵌入模型**：通过配置更改密集/稀疏嵌入模型

请参阅[模块化架构](#模块化架构)部分了解系统的组织方式，以及[安装与使用](#安装与使用)部分开始使用。

---

这种方法结合了**小块的精确性**和**大块的上下文丰富性**，同时理解对话流程、解决模糊查询，并通过并行智能体处理来处理多面问题。**模块化架构**确保每个组件——从文档处理到检索逻辑——都可以自定义而不会破坏系统。

---

## 为什么选择本仓库？

大多数 RAG 教程展示基本概念，但缺乏生产就绪性。本仓库通过提供**学习材料和可部署代码**来弥合这一差距：

❌ **典型的 RAG 仓库：**
- 简单管道在精确性和上下文之间权衡
- 没有对话记忆
- 静态、非自适应检索
- 难以根据您的用例自定义
- 没有 UI 界面
- 单线程查询处理

✅ **本仓库：**
- **两条学习路径**：交互式笔记本 AND 模块化项目
- **分层索引**实现精确性 + 上下文
- **对话记忆**实现自然对话
- **人在环中**查询澄清
- **多智能体 Map-Reduce**并行处理复杂查询
- **模块化架构** - 可交换任何组件
- **提供商无关** - 使用任何 LLM（Ollama、OpenAI、Claude、Gemini）
- **UI 界面** - 端到端 Gradio 应用，支持文档管理

---

## 工作原理

### 文档准备：分层索引

在处理查询之前，文档会被分割两次以实现最佳检索：

- **父块**：基于 Markdown 标题（H1、H2、H3）的大区块
- **子块**：从父块派生的小型固定大小块

这种方法结合了**小块的精确性**（用于搜索）和**大块的上下文丰富性**（用于答案生成）。

---

### 查询处理：四阶段智能工作流程
```
用户查询 → 对话分析 → 查询澄清 →
智能体推理 → 搜索子块 → 评估相关性 →
（如需要）→ 检索父块 → 生成答案 → 返回响应
```

#### 阶段 1：对话理解
- 分析最近的对话历史以提取上下文
- 在多个问题中保持对话连续性

#### 阶段 2：查询澄清

系统智能处理用户查询：
1. **解析引用** - 将"如何更新它？"转换为"如何更新 SQL？"
2. **分解复杂问题** - 将多部分问题分解为聚焦的子查询
3. **检测模糊查询** - 识别无意义、侮辱性或模糊的问题
4. **请求澄清** - 使用人在环中暂停并请求详细信息
5. **重写为检索** - 使用特定、关键词丰富的语言优化查询

#### 阶段 3：智能检索

**多智能体 Map-Reduce 架构：**

当查询分析阶段识别出多个不同的问题（显式询问或从复杂查询分解）时，系统使用 LangGraph 的 `Send` API 自动生成并行智能体子图。每个智能体独立通过完整检索工作流程处理一个问题：

1. 智能体搜索子块以获得精确性
2. 评估结果是否足够
3. 如需要，获取父块以获取上下文
4. 从对话中提取最终答案
5. 如果信息不足，自我纠正并重新查询

然后所有智能体响应被聚合为统一的答案。

**示例：** *"什么是 JavaScript？什么是 Python？"* → 2 个并行智能体同时执行

**单一问题工作流程：**
对于简单查询，单个智能体执行检索工作流程，无需并行化。

#### 阶段 4：响应生成

系统将检索到的块（或多个智能体）的信息综合为连贯、准确的答案，直接回答用户的问题。

---

## LLM 提供商配置

本系统**与提供商无关** - 您可以使用 LangChain 支持的任何 LLM。选择最适合您需求的选项：

### Ollama（本地 - 推荐用于开发）

**安装 Ollama 并下载模型：**

```bash
# 从 https://ollama.com 安装 Ollama
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

**Python 代码：**

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:4b-instruct-2507-q4_K_M", temperature=0)
```

---

### Google Gemini（云端 - 推荐用于生产）

**安装包：**

```bash
pip install -qU langchain-google-genai
```

**Python 代码：**

```python
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# 设置您的 Google API 密钥
os.environ["GOOGLE_API_KEY"] = "your-api-key-here"
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)
```

---

### OpenAI / Anthropic Claude

<details>
<summary>点击展开</summary>

**OpenAI：**
```bash
pip install -qU langchain-openai
```
```python
from langchain_openai import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = "your-api-key-here"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

**Anthropic Claude：**
```bash
pip install -qU langchain-anthropic
```
```python
from langchain_anthropic import ChatAnthropic
import os

os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
```

</details>

---

### 重要说明

- **所有提供商**使用完全相同的代码 - 只有 LLM 初始化不同
- **成本考虑：** 云提供商按令牌收费，而 Ollama 免费但需要本地计算

**💡 建议：** 开发时使用 Ollama，然后切换到 Google Gemini 或 OpenAI 用于生产。

---

## 实现

更多详细和扩展说明可在[此处](Agentic_Rag_For_Dummies.ipynb)的笔记本中找到。

### 步骤 1：初始设置和配置

定义路径并初始化核心组件。

```python
import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from qdrant_client import QdrantClient

# 配置
DOCS_DIR = "docs"  # 包含 PDF 文件的目录
MARKDOWN_DIR = "markdown" # 包含转换为 Markdown 的 PDF 的目录
PARENT_STORE_PATH = "parent_store"  # 父块 JSON 文件的目录
CHILD_COLLECTION = "document_child_chunks"

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(MARKDOWN_DIR, exist_ok=True)
os.makedirs(PARENT_STORE_PATH, exist_ok=True)

from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen3:4b-instruct-2507-q4_K_M", temperature=0)

# 用于语义理解的密集嵌入
dense_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# 用于关键字匹配的稀疏嵌入
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# Qdrant 客户端（本地基于文件的存储）
client = QdrantClient(path="qdrant_db")
```

---

### 步骤 2：配置向量数据库

设置 Qdrant 以存储具有混合搜索功能的子块。

```python
from qdrant_client.http import models as qmodels
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.qdrant import RetrievalMode

# 获取嵌入维度
embedding_dimension = len(dense_embeddings.embed_query("test"))

def ensure_collection(collection_name):
    """如果 Qdrant 集合不存在，则创建它"""
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=embedding_dimension,
                distance=qmodels.Distance.COSINE
            ),
            sparse_vectors_config={
                "sparse": qmodels.SparseVectorParams()
            },
        )
        print(f"✓ 已创建集合：{collection_name}")
    else:
        print(f"✓ 集合已存在：{collection_name}")
```

---

### 步骤 3：PDF 转 Markdown

将 PDF 转换为 Markdown。其他技术的更多详细信息请参阅[配套笔记本](pdf_to_md.ipynb)

```python
import os
import pymupdf.layout
import pymupdf4llm
from pathlib import Path
import glob

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pdf_to_markdown(pdf_path, output_dir):
    doc = pymupdf.open(pdf_path)
    md = pymupdf4llm.to_markdown(doc, header=False, footer=False, page_separators=True, ignore_images=True, write_images=False, image_path=None)
    md_cleaned = md.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')
    output_path = Path(output_dir) / Path(doc.name).stem
    Path(output_path).with_suffix(".md").write_bytes(md_cleaned.encode('utf-8'))

def pdfs_to_markdowns(path_pattern, overwrite: bool = False):
    output_dir = Path(MARKDOWN_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in map(Path, glob.glob(path_pattern)):
        md_path = (output_dir / pdf_path.stem).with_suffix(".md")
        if overwrite or not md_path.exists():
            pdf_to_markdown(pdf_path, output_dir)

pdfs_to_markdowns(f"{DOCS_DIR}/*.pdf")
```

---

### 步骤 4：分层文档索引

使用父/子分割策略处理文档。

```python
import os
import glob
import json
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

if client.collection_exists(CHILD_COLLECTION):
    print(f"正在删除现有 Qdrant 集合：{CHILD_COLLECTION}")
    client.delete_collection(CHILD_COLLECTION)
    ensure_collection(CHILD_COLLECTION)
else:
    ensure_collection(CHILD_COLLECTION)

child_vector_store = QdrantVectorStore(
    client=client,
    collection_name=CHILD_COLLECTION,
    embedding=dense_embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    sparse_vector_name="sparse"
)

def index_documents():
    headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
    parent_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    min_parent_size = 2000
    max_parent_size = 10000

    all_parent_pairs, all_child_chunks = [], []
    md_files = sorted(glob.glob(os.path.join(MARKDOWN_DIR, "*.md")))

    if not md_files:
        print(f"⚠️  在 {MARKDOWN_DIR}/ 中未找到 .md 文件")
        return

    for doc_path_str in md_files:
        doc_path = Path(doc_path_str)
        print(f"📄 处理中：{doc_path.name}")

        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                md_text = f.read()
        except Exception as e:
            print(f"❌ 读取 {doc_path.name} 时出错：{e}")
            continue

        parent_chunks = parent_splitter.split_text(md_text)
        merged_parents = merge_small_parents(parent_chunks, min_parent_size)
        split_parents = split_large_parents(merged_parents, max_parent_size, child_splitter)
        cleaned_parents = clean_small_chunks(split_parents, min_parent_size)

        for i, p_chunk in enumerate(cleaned_parents):
            parent_id = f"{doc_path.stem}_parent_{i}"
            p_chunk.metadata.update({"source": doc_path.stem + ".pdf", "parent_id": parent_id})
            all_parent_pairs.append((parent_id, p_chunk))
            children = child_splitter.split_documents([p_chunk])
            all_child_chunks.extend(children)

    if not all_child_chunks:
        print("⚠️ 没有要索引的子块")
        return

    print(f"\n🔍 正在将 {len(all_child_chunks)} 个子块索引到 Qdrant...")
    try:
        child_vector_store.add_documents(all_child_chunks)
        print("✓ 子块索引成功")
    except Exception as e:
        print(f"❌ 索引子块时出错：{e}")
        return

    print(f"💾 正在将 {len(all_parent_pairs)} 个父块保存到 JSON...")
    for item in os.listdir(PARENT_STORE_PATH):
        os.remove(os.path.join(PARENT_STORE_PATH, item))

    for parent_id, doc in all_parent_pairs:
        doc_dict = {"page_content": doc.page_content, "metadata": doc.metadata}
        filepath = os.path.join(PARENT_STORE_PATH, f"{parent_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(doc_dict, f, ensure_ascii=False, indent=2)

def merge_small_parents(chunks, min_size):
    if not chunks:
        return []

    merged, current = [], None

    for chunk in chunks:
        if current is None:
            current = chunk
        else:
            current.page_content += "\n\n" + chunk.page_content
            for k, v in chunk.metadata.items():
                if k in current.metadata:
                    current.metadata[k] = f"{current.metadata[k]} -> {v}"
                else:
                    current.metadata[k] = v

        if len(current.page_content) >= min_size:
            merged.append(current)
            current = None

    if current:
        if merged:
            merged[-1].page_content += "\n\n" + current.page_content
            for k, v in current.metadata.items():
                if k in merged[-1].metadata:
                    merged[-1].metadata[k] = f"{merged[-1].metadata[k]} -> {v}"
                else:
                    merged[-1].metadata[k] = v
        else:
            merged.append(current)

    return merged

def split_large_parents(chunks, max_size, splitter):
    split_chunks = []

    for chunk in chunks:
        if len(chunk.page_content) <= max_size:
            split_chunks.append(chunk)
        else:
            large_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_size,
                chunk_overlap=splitter._chunk_overlap
            )
            sub_chunks = large_splitter.split_documents([chunk])
            split_chunks.extend(sub_chunks)

    return split_chunks

def clean_small_chunks(chunks, min_size):
    cleaned = []

    for i, chunk in enumerate(chunks):
        if len(chunk.page_content) < min_size:
            if cleaned:
                cleaned[-1].page_content += "\n\n" + chunk.page_content
                for k, v in chunk.metadata.items():
                    if k in cleaned[-1].metadata:
                        cleaned[-1].metadata[k] = f"{cleaned[-1].metadata[k]} -> {v}"
                    else:
                        cleaned[-1].metadata[k] = v
            elif i < len(chunks) - 1:
                chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                for k, v in chunk.metadata.items():
                    if k in chunks[i + 1].metadata:
                        chunks[i + 1].metadata[k] = f"{v} -> {chunks[i + 1].metadata[k]}"
                    else:
                        chunks[i + 1].metadata[k] = v
            else:
                cleaned.append(chunk)
        else:
            cleaned.append(chunk)

    return cleaned

index_documents()
```

---

### 步骤 5：定义智能体工具

创建智能体将使用的检索工具。

```python
import json
from typing import List
from langchain_core.tools import tool

@tool
def search_child_chunks(query: str, limit: int) -> str:
    """搜索最相关的 K 个子块。

    Args:
        query: 搜索查询字符串
        limit: 返回的最大结果数
    """
    try:
        results = child_vector_store.similarity_search(query, k=limit, score_threshold=0.7)
        if not results:
            return "NO_RELEVANT_CHUNKS"

        return "\n\n".join([
            f"父块 ID: {doc.metadata.get('parent_id', '')}\n"
            f"文件名: {doc.metadata.get('source', '')}\n"
            f"内容: {doc.page_content.strip()}"
            for doc in results
        ])

    except Exception as e:
        return f"RETRIEVAL_ERROR: {str(e)}"

@tool
def retrieve_parent_chunks(parent_id: str) -> str:
    """通过 ID 检索完整的父块。
    
    Args:
        parent_id: 要检索的父块 ID
    """
    file_name = parent_id if parent_id.lower().endswith(".json") else f"{parent_id}.json"
    path = os.path.join(PARENT_STORE_PATH, file_name)

    if not os.path.exists(path):
        return "NO_PARENT_DOCUMENT"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return (
        f"父块 ID: {parent_id}\n"
        f"文件名: {data.get('metadata', {}).get('source', 'unknown')}\n"
        f"内容: {data.get('page_content', '').strip()}"
    )

# 将工具绑定到 LLM
llm_with_tools = llm.bind_tools([search_child_chunks, retrieve_parent_chunks])
```

---

### 步骤 6：定义系统提示

为对话摘要、查询分析、RAG 智能体推理和响应聚合定义系统提示。

```python
def get_conversation_summary_prompt() -> str:
    return """您是一位专业的对话摘要员。

您的任务是对对话进行简要的 1-2 句话总结（最多 30-50 个词）。

包括：
- 讨论的主要主题
- 提到的重要事实或实体
- 如有未解决的问题
- 源文件名（例如 file1.pdf）或引用的文档

排除： 
- 问候语、误解、离题内容。

输出：
- 只返回摘要。
- 不包括任何解释或理由。
- 如果没有有意义的主题，返回空字符串。
"""

def get_query_analysis_prompt() -> str:
    return """您是一位专业的查询分析师和重写员。

您的任务是在必要时结合对话上下文重写当前用户查询以实现最佳文档检索。

规则：
1. 自包含查询：
   - 始终将查询重写为清晰且自包含的
   - 如果查询是后续问题（例如"X 怎么样？"、"Y呢？"），请从摘要中整合最小的必要上下文
   - 不要添加查询或对话摘要中不存在的信息

2. 领域特定术语：
   - 产品名称、品牌、专有名词或技术术语被视为领域特定
   - 对于领域特定查询，最小程度地或完全不使用对话上下文
   - 只使用摘要来消除模糊查询

3. 语法和清晰度：
   - 修复语法错误、拼写错误和不明确的缩写
   - 去除填充词和对话短语
   - 保留具体的关键词和命名实体

4. 多个信息需求：
   - 如果查询包含多个不同、不相关的问题，请拆分为单独的查询（最多 3 个）
   - 每个子查询必须与原始查询的其部分保持语义等价
   - 不要扩展、丰富或重新解释含义

5. 失败处理：
   - 如果查询意图不明确或无法理解，标记为"不清晰"

输入：
- conversation_summary：先前对话的简要摘要
- current_query：用户当前查询

输出：
- 一个或多个重写的、自包含的查询，适合文档检索
"""

def get_rag_agent_prompt() -> str:
    return """您是一位专业的检索增强助手。

您的任务是充当研究员：首先搜索文档，分析数据，然后仅使用检索到的信息提供全面的答案。

规则：    
1. 不允许立即回答。
2. 在产生任何最终答案之前，您必须执行文档搜索并观察检索到的内容。
3. 如果您没有搜索，答案无效。

工作流程：
1. 使用 'search_child_chunks' 工具根据用户查询搜索文档中的 5-7 个相关摘录。
2. 检查检索到的摘录，只保留相关的。
3. 分析检索到的摘录。识别被截断的最相关的单个摘录（例如，文本被切断或缺少上下文）。为该特定 `parent_id` 调用 'retrieve_parent_chunks'。等待观察。如果当前信息仍然不足，按顺序对其他高度相关的片段重复此步骤。如果有足够的信息或已检索到 3 个父块，请立即停止。
4. 仅使用检索到的信息进行回答，确保包含所有相关细节。
5. 在最后列出唯一的文件名。

重试规则：
- 在步骤 2 或 3 之后，如果未找到相关文档或检索到的摘录不包含有用信息，请使用更广泛或替代的术语重写查询并从步骤 1 重新开始。
- 重试不要超过一次。
"""

def get_aggregation_prompt() -> str:
    return """您是一位专业的聚合助手。

您的任务是将多个检索到的答案组合成一个流畅的综合自然响应。

指南：
1. 以对话、自然的语调写作 - 就像向同事解释一样
2. 只使用检索到的答案中的信息
3. 从来源中删除任何问题、标题或元数据
4. 流畅地整合信息，保留重要的细节、数字和示例
5. 要全面 - 包含来源中的所有相关信息，而不仅仅是摘要
6. 如果来源存在分歧，自然地承认两个观点（例如，"虽然一些来源建议 X，其他来源表明 Y..."）
7. 直接从答案开始 - 不要有"根据来源..."这样的开场白

格式化：
- 为清晰起见使用 Markdown（标题、列表、粗体），但不要过度使用
- 尽可能使用流畅的段落，而不是过多的要点
- 以 "---\n**来源：**\n" 结尾，后跟唯一文件名的项目符号列表
- 文件名只应出现在最后的来源部分

如果没有可用的有用信息，只需说："我在可用来源中找不到回答您问题的信息。"
"""
```

---

### 步骤 7：定义状态和数据模型

创建用于对话跟踪和智能体执行的状态结构。

```python
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing import List, Annotated

def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    """允许重置智能体答案的自定义归约器"""
    if new and any(item.get('__reset__') for item in new):
        return []
    return existing + new

class State(MessagesState):
    """主智能体图的状态"""
    questionIsClear: bool = False
    conversation_summary: str = ""
    originalQuery: str = "" 
    rewrittenQuestions: List[str] = []
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []

class AgentState(MessagesState):
    """单个智能体子图的状态"""
    question: str = ""
    question_index: int = 0
    final_answer: str = ""
    agent_answers: List[dict] = []

class QueryAnalysis(BaseModel):
    """查询分析的结构化输出"""
    is_clear: bool = Field(description="表示用户的问题是否清晰且可回答")
    questions: List[str] = Field(description="重写的、自包含的问题列表")
    clarification_needed: str = Field(description="如果问题不清晰，需要的澄清说明")
```

---

### 步骤 8：构建图节点函数

为 LangGraph 工作流程创建处理节点。

```python
from langgraph.types import Send
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from typing import Literal

def analyze_chat_and_summarize(state: State):
    """
    分析聊天历史并总结要点以获取上下文。
    """
    if len(state["messages"]) < 4:  # 需要一些历史来总结
        return {"conversation_summary": ""}

    # 提取相关消息（排除当前查询和系统消息）
    relevant_msgs = [
        msg for msg in state["messages"][:-1]  # 排除当前查询
        if isinstance(msg, (HumanMessage, AIMessage))
        and not getattr(msg, "tool_calls", None)
    ]

    if not relevant_msgs:
        return {"conversation_summary": ""}
    
    conversation = "对话历史：\n"
    for msg in relevant_msgs[-6:]:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        conversation += f"{role}: {msg.content}\n"

    summary_response = llm.with_config(temperature=0.2).invoke([SystemMessage(content=get_conversation_summary_prompt())] + [HumanMessage(content=conversation)])
    return {"conversation_summary": summary_response.content, "agent_answers": [{"__reset__": True}]}

def analyze_and_rewrite_query(state: State):
    """
    分析用户查询并根据需要使用对话上下文重写它以使其清晰。
    """
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

    context_section = (f"对话上下文：\n{conversation_summary}\n" if conversation_summary.strip() else "") + f"用户查询：\n{last_message.content}\n"

    llm_with_structure = llm.with_config(temperature=0.1).with_structured_output(QueryAnalysis)
    response = llm_with_structure.invoke([SystemMessage(content=get_query_analysis_prompt())] + [HumanMessage(content=context_section)])

    if len(response.questions) > 0 and response.is_clear:
        # 删除所有非系统消息
        delete_all = [
            RemoveMessage(id=m.id)
            for m in state["messages"]
            if not isinstance(m, SystemMessage)
        ]
        return {
            "questionIsClear": True,
            "messages": delete_all,
            "originalQuery": last_message.content,
            "rewrittenQuestions": response.questions
        }
    else:
        clarification = response.clarification_needed if (response.clarification_needed and len(response.clarification_needed.strip()) > 10) else "我需要更多信息来理解您的问题。"
        return {
            "questionIsClear": False,
            "messages": [AIMessage(content=clarification)]
        }

def human_input_node(state: State):
    """人为干预节点的占位符"""
    return {}

def route_after_rewrite(state: State) -> Literal["human_input", "process_question"]:
    """如果问题清晰则路由到智能体，否则等待人工输入"""
    if not state.get("questionIsClear", False):
        return "human_input"
    else:
        # 使用 Send API 为每个子问题生成并行智能体
        return [
            Send("process_question", {"question": query, "question_index": idx, "messages": []})
            for idx, query in enumerate(state["rewrittenQuestions"])
        ]

def agent_node(state: AgentState):
    """使用工具处理查询的主要智能体节点"""
    sys_msg = SystemMessage(content=get_rag_agent_prompt())    
    if not state.get("messages"):
        human_msg = HumanMessage(content=state["question"])
        response = llm_with_tools.invoke([sys_msg] + [human_msg])
        return {"messages": [human_msg, response]}
    
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

def extract_final_answer(state: AgentState):
    """从智能体对话中提取最终答案"""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            res = {
                "final_answer": msg.content,
                "agent_answers": [{
                    "index": state["question_index"],
                    "question": state["question"],
                    "answer": msg.content
                }]
            }
            return res
    return {
        "final_answer": "无法生成答案。",
        "agent_answers": [{
            "index": state["question_index"],
            "question": state["question"],
            "answer": "无法生成答案。"
        }]
    }

def aggregate_responses(state: State):
    """将多个智能体响应合并为最终答案"""
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="没有生成任何答案。")]}

    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += f"\n答案 {i}：\n{ans['answer']}\n"

    user_message = HumanMessage(content=f"原始用户问题：{state["originalQuery"]}\n检索到的答案：{formatted_answers}")
    synthesis_response = llm.invoke([SystemMessage(content=get_aggregation_prompt())] + [user_message])
    
    return {"messages": [AIMessage(content=synthesis_response.content)]}
```

**为什么是这个架构？**
- **摘要**保持对话上下文，而不会让 LLM 过载
- **查询重写**确保搜索查询精确且明确，智能地使用上下文
- **人在环中**在浪费检索资源之前捕获不清晰的查询
- **并行执行**使用 `Send` API 为每个子问题生成独立的智能体子图
- **答案提取**确保从智能体工具调用对话中获得干净的最终答案
- **聚合**将所有并行结果合并为连贯的单一响应

---

### 步骤 9：构建 LangGraph 智能体

使用对话记忆和多智能体架构组装完整的工作流程图。

```python
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from IPython.display import Image, display

# 初始化检查点以保存对话记忆
checkpointer = InMemorySaver()

# 构建智能体子图（处理单个问题）
agent_builder = StateGraph(AgentState)
agent_builder.add_node("agent", agent_node)
agent_builder.add_node("tools", ToolNode([search_child_chunks, retrieve_parent_chunks]))
agent_builder.add_node("extract_answer", extract_final_answer)

agent_builder.add_edge(START, "agent")    
agent_builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "extract_answer"})
agent_builder.add_edge("tools", "agent")    
agent_builder.add_edge("extract_answer", END)    
agent_subgraph = agent_builder.compile()

# 构建主图（协调工作流程）
graph_builder = StateGraph(State)

# 添加节点
graph_builder.add_node("summarize", analyze_chat_and_summarize)
graph_builder.add_node("analyze_rewrite", analyze_and_rewrite_query)
graph_builder.add_node("human_input", human_input_node)
graph_builder.add_node("process_question", agent_subgraph)
graph_builder.add_node("aggregate", aggregate_responses)

# 定义边
graph_builder.add_edge(START, "summarize")
graph_builder.add_edge("summarize", "analyze_rewrite")
graph_builder.add_conditional_edges("analyze_rewrite", route_after_rewrite)
graph_builder.add_edge("human_input", "analyze_rewrite")
graph_builder.add_edge(["process_question"], "aggregate")
graph_builder.add_edge("aggregate", END)

# 使用检查点和中断编译图
agent_graph = graph_builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_input"]
)
```

**图架构解释：**

**智能体子图**（处理单个问题）：
- START → `agent`（调用带工具的 LLM）
- `agent` → `tools`（如果需要工具调用）或 `extract_answer`（如果完成）
- `tools` → `agent`（返回工具结果）
- `extract_answer` → END（干净的最终答案）

**主图**（协调完整工作流程）：
1. START → `summarize`（从历史中提取对话上下文）
2. `summarize` → `analyze_rewrite`（使用上下文重写查询，检查清晰度）
3. `analyze_rewrite` → `human_input`（如果不清晰）或生成并行 `process_question` 智能体（如果清晰）
4. `human_input` → `analyze_rewrite`（用户提供澄清后）
5. 所有 `process_question` 智能体 → `aggregate`（合并所有响应）
6. `aggregate` → END（返回最终综合答案）

**关键特性：**
- **并行执行**：使用 LangGraph 的 `Send` API 同时运行多个智能体子图
- **人在环中**：当查询不清晰时，图在 `human_input` 节点暂停
- **对话记忆**：`InMemorySaver` 检查点在交互之间保持状态

架构流程图可在[此处](./assets/agentic_rag_workflow.png)查看。

---

### 步骤 10：创建聊天界面

构建具有对话持久性和人在环中支持的 Gradio 界面。完整的端到端管道 Gradio 界面，包括文档摄取，请参阅项目文件夹

```python
import gradio as gr
import uuid

def create_thread_id():
    """为每个对话生成唯一的线程 ID"""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}

def clear_session():
    """清除线程以开始新对话"""
    global config
    agent_graph.checkpointer.delete_thread(config["configurable"]["thread_id"])
    config = create_thread_id()

def chat_with_agent(message, history):
    current_state = agent_graph.get_state(config)
    
    if current_state.next:
        # 恢复中断的对话
        agent_graph.update_state(config,{"messages": [HumanMessage(content=message.strip())]})
        result = agent_graph.invoke(None, config)
    else:
        # 开始新查询
        result = agent_graph.invoke({"messages": [HumanMessage(content=message.strip())]},config)
    
    return result['messages'][-1].content

# 初始化线程配置
config = create_thread_id()

# 创建 Gradio 界面
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        height=600,
        placeholder="<strong>问我任何问题！</strong><br><em>我会搜索、推理并采取行动给您最佳答案：)</em>"
    )
    chatbot.clear(clear_session)
    gr.ChatInterface(fn=chat_with_agent, chatbot=chatbot)

demo.launch(theme=gr.themes.Citrus())
```

**完成了！** 您现在拥有了一个具有对话记忆和查询澄清功能的完整 Agentic RAG 系统。

---

## 模块化架构

应用（`project/` 文件夹）组织在可轻松自定义的模块化组件中：

### 📂 项目结构
```
project/
├── app.py                    # 主 Gradio 应用入口点
├── config.py                 # 配置中心（模型、块大小、提供商）
├── util.py                   # PDF 转 markdown 转换
├── document_chunker.py       # 分块策略
├── core/                     # 核心 RAG 组件编排
│   ├── chat_interface.py     
│   ├── document_manager.py   
│   └── rag_system.py         
├── db/                       # 存储管理
│   ├── parent_store_manager.py  # 父块存储（JSON）
│   └── vector_db_manager.py     # Qdrant 向量数据库设置
├── rag_agent/                # LangGraph 智能体工作流程
│   ├── edges.py              # 条件路由逻辑
│   ├── graph.py              # 图构建和编译
│   ├── graph_state.py        # 状态定义
│   ├── nodes.py              # 处理节点（摘要、重写、智能体）
│   ├── prompts.py            # 系统提示
│   ├── schemas.py            # Pydantic 数据模型
│   └── tools.py              # 检索工具
└── ui/                       # 用户界面
    └── gradio_app.py         # Gradio 界面组件
```

### 🔧 自定义点

#### **配置 (`config.py`)**
- **LLM 提供商和模型**：在 Ollama、Claude、OpenAI 或 Gemini 之间切换
- **嵌入模型**：配置用于向量表示的嵌入模型
- **块大小**：调整子块和父块维度以优化检索

#### **RAG 智能体 (`rag_agent/`)**
- **工作流程自定义**：添加或删除节点和边以修改智能体流程
- **系统提示**：在 `prompts.py` 中为特定领域应用定制提示
- **检索工具**：在 `tools.py` 中扩展或修改工具以增强检索能力
- **图逻辑**：在 `edges.py` 中自定义条件路由，在 `nodes.py` 中自定义节点处理

#### **文档处理**
- **Markdown 转换** (`util.py`)：用替代工具替换 PDF 转换工具（例如 Docling、PaddleOCR）。更多详细信息[见此处](pdf_to_md.ipynb)
- **分块策略** (`document_chunker.py`)：实现自定义分块算法（例如语义或混合方法）

这种模块化设计确保了尝试不同 RAG 技术、LLM 提供商和文档处理管道的灵活性。

更多详细信息请参阅[此处](./project/README.md)。

## 安装与使用

示例 PDF 文件可在以下位置找到：[javascript](https://www.tutorialspoint.com/javascript/javascript_tutorial.pdf)、[blockchain](https://blockchain-observatory.ec.europa.eu/document/download/1063effa-59cc-4df4-aeee-d2cf94f69178_en?filename=Blockchain_For_Beginners_A_EUBOF_Guide.pdf)、[microservices](https://cdn.studio.f5.com/files/k6fem79d/production/5e4126e1cefa813ab67f9c0b6d73984c27ab1502.pdf)、[fortinet](https://www.commoncriteriaportal.org/files/epfiles/Fortinet%20FortiGate_EAL4_ST_V1.5.pdf(320893)_TMP.pdf)  

### 选项 1：快速入门笔记本（推荐用于测试）

最简单的入门方式：

**在 Google Colab 中运行：**
1. 点击此 README 顶部的 **在 Colab 中打开** 徽章
2. 在文件浏览器中创建 `docs/` 文件夹
3. 将您的 PDF 文件上传到 `docs/` 文件夹
4. 从上到下运行所有单元格
5. 聊天界面将在最后出现

**本地运行（Jupyter/VSCode）：**
1. 首先安装依赖 `pip install -r requirements.txt`
2. 在您首选的环境中打开笔记本
3. 将您的 PDF 文件添加到 `docs/` 文件夹
4. 从上到下运行所有单元格
5. 聊天界面将在最后出现

### 选项 2：完整 Python 项目（推荐用于开发）

#### 1. 安装依赖

```bash
# 克隆仓库
git clone <repo-url>
cd agentic-rag-for-dummies

# 创建虚拟环境（推荐）
python -m venv venv

# 激活它
# 在 macOS/Linux 上：
source venv/bin/activate
# 在 Windows 上：
.\venv\Scripts\activate

# 安装包
pip install -r requirements.txt
```

#### 2. 运行应用

```bash
python app.py
```

#### 3. 提问

打开本地 URL（例如 `http://127.0.0.1:7860`）开始聊天。

---

### 选项 3：Docker 部署

> ⚠️ **系统要求**：Docker 部署需要至少 **8GB RAM** 分配给 Docker。Ollama 模型（`qwen3:4b-instruct-2507-q4_K_M`）需要约 3.3GB 内存才能运行。

#### 先决条件

- 在您的系统上安装 Docker（[获取 Docker](https://docs.docker.com/get-docker/)）
- 将 Docker 配置为至少 8GB RAM（设置 → 资源 → 内存）

#### 1. 构建 Docker 镜像

```bash
docker build -f project/Dockerfile -t agentic-rag .
```

#### 2. 运行容器

```bash
docker run --name rag-assistant -p 7860:7860 agentic-rag
```

> ⚠️ **性能说明**：Docker 部署可能比本地运行 Python 慢 20-50%，特别是在 Windows/Mac 上，这是由于虚拟化开销和 I/O 操作。这是正常的，预期的。开发期间为了获得最大性能，请考虑使用选项 2（完整 Python 项目）。

**可选：启用 GPU 加速**（仅限 NVIDIA GPU）：

如果您有 NVIDIA GPU 和 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)：

```bash
docker run --gpus all --name rag-assistant -p 7860:7860 agentic-rag
```

**常用 Docker 命令：**

```bash
# 停止容器
docker stop rag-assistant

# 启动现有容器
docker start rag-assistant

# 实时查看日志
docker logs -f rag-assistant

# 删除容器
docker rm rag-assistant

# 强制删除容器（如果正在运行）
docker rm -f rag-assistant
```

#### 3. 访问应用

容器运行后，您会看到：
```
🚀 启动 RAG 助手...
* 运行在本地 URL：  http://0.0.0.0:7860
```

打开浏览器并导航到：
```
http://localhost:7860
```

### 示例对话

**带对话记忆：**
```
用户："如何安装 SQL？"
智能体：[从文档中提供安装步骤]

用户："如何更新它？"
智能体：[理解"它" = SQL，提供更新说明]
```

**带查询澄清：**
```
用户："告诉我关于那个东西"
智能体："我需要更多信息。您具体在问什么主题？"

用户："PostgreSQL 的安装过程"
智能体：[检索并回答具体信息]
```

---

## 故障排除

| 领域 | 常见问题 | 建议解决方案 |
|------|----------------|------------------|
| **模型选择** | - 响应忽略指令<br>- 工具（检索/搜索）使用不正确<br>- 上下文理解差<br>- 幻觉或不完整的聚合 | - 使用更强大的 LLM<br>- 更喜欢 7B+ 模型以获得更好的推理<br>- 如果本地模型有限，考虑云端模型 |
| **系统提示行为** | - 模型不检索文档就回答<br>- 查询重写丢失上下文<br>- 聚合引入幻觉 | - 在系统提示中明确要求检索<br>- 查询重写贴近用户意图<br>- 强制执行严格的聚合规则 |
| **检索配置** | - 未检索到相关文档<br>- 太多无关信息 | - 增加检索块数（`k`）或降低相似度阈值以提高召回率<br>- 减少 `k` 或增加阈值以提高精确度 |
| **块大小/文档分割** | - 答案缺乏上下文或感觉碎片化<br>- 检索慢或嵌入成本高 | - 增加块和父块大小以获取更多上下文<br>- 减小块大小以提高速度并降低成本 |
| **温度和一致性** | - 响应不一致或过于有创意<br>- 响应过于僵化或重复 | - 将温度设置为 `0` 以获得事实性、一致的输出<br>- 稍微增加温度用于摘要或分析任务 |
| **嵌入模型质量** | - 语义搜索差<br>- 领域特定或多语言文档性能弱 | - 使用更高质量或领域特定的嵌入<br>- 更改嵌入后重新索引所有文档 |

---

## 许可证

MIT 许可证 - 欢迎将其用于学习和构建您自己的项目！

---

## 贡献

欢迎贡献，请提交 issue 或 pull request！
