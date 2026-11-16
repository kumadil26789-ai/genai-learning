# Document Q&A with GenAI

A beginner-friendly Retrieval-Augmented Generation (RAG) project. Upload PDF or
text files, find the passages most relevant to a question, and ask an OpenAI
model to answer using those passages with source references.

## Step-by-step

### 1. Create a Python environment

Use Python 3.10 or later:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Configure your OpenAI API key

Copy `.env.example` to `.env` and add your API key:

```text
OPENAI_API_KEY=your_api_key_here
```

Do not commit `.env`. The contents of uploaded documents are sent to OpenAI's
embedding and chat APIs when you index them and ask questions.

### 3. Start the app

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, select one or more PDF or TXT files,
and choose **Index documents**. Enter a question after indexing.

### 4. Follow the RAG pipeline

1. **Load:** `rag.py` extracts text from each PDF page or TXT file.
2. **Chunk:** long text is split into overlapping passages so context is small
   enough to retrieve and send to the model.
3. **Embed:** OpenAI's `text-embedding-3-small` model turns each passage into a
   vector that represents its meaning.
4. **Retrieve:** the question is embedded too; cosine similarity selects the
   four closest passages.
5. **Generate:** the chat model receives the question and retrieved passages.
   The prompt asks it to stick to that context and cite the source labels.

This starter app keeps vectors in Streamlit's session state; they are not saved
between sessions and no vector database is required.

### 5. Run the tests

The unit tests cover chunking and similarity-based retrieval without making
OpenAI API calls:

```powershell
python -m unittest discover -s tests -v
```

## Project files

- `app.py` — Streamlit interface and API workflow
- `rag.py` — extraction, chunking, embeddings, retrieval, and answer generation
- `tests/test_rag.py` — offline unit tests
- `requirements.txt` — project dependencies

## Troubleshooting

- If indexing reports a missing API key, confirm `.env` exists beside `app.py`
  and contains `OPENAI_API_KEY=...`.
- Scanned/image-only PDFs need OCR; this starter project extracts selectable
  PDF text only.
- An API/network error is shown in the app. Check your API access and connection,
  then try indexing again.
