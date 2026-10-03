# AI Teacher: PDF-to-vector database

A small learning project that turns UPSC source PDFs into separate local ChromaDB databases. It intentionally stops at ingestion and retrieval: no LLM examiner or answer-evaluation system is included.

## What it does

`PDF -> PyMuPDF text -> chapter/section parsing -> semantic chunks -> sentence-transformer embeddings -> ChromaDB`

- Uses the PDF bookmark/outline hierarchy when available, then falls back to visible Contents and numbered body headings.
- Recognises chapter labels from text (for example, `7 Fundamental Rights`) rather than treating PDF page numbers as chapters.
- Uses lower-level bookmark labels as section/topic names.
- Groups paragraphs into readable chunks and only splits an oversized paragraph at sentence boundaries; it does not slice text at fixed character offsets.
- Stores every chunk's subject, book, chapter, section, ID, PDF page, and text as Chroma metadata.
- Creates one persistent ChromaDB directory per book, so sources do not become one oversized mixed database.

## Project structure

```text
AI Teacher/
├── data/
│   └── books/
│       ├── polity/laxmikanth.pdf
│       ├── economics/ncert_class_11.pdf
│       └── economics/ncert_class_12.pdf
├── vector_dbs/                  # generated: one ChromaDB directory per book
│   ├── polity/laxmikanth/
│   ├── economics/ncert_class_11/
│   └── book_catalog.json        # maps subjects/books to DB directories
├── pdf_extractor.py             # text and bookmark extraction
├── structure_parser.py          # chapters and sections
├── chunker.py                   # sentence/paragraph-aware chunking
├── embeddings.py                # Sentence Transformers wrapper
├── vector_store.py              # persistent ChromaDB storage
├── retriever.py                 # retrieval API
├── main.py                      # ingest/query command-line interface
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.10 or later. Create and activate a virtual environment, then install the packages listed in `requirements.txt`:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`sentence-transformers` downloads the default embedding model (`all-MiniLM-L6-v2`) on its first use. Keep an internet connection available for that first run.

Put text-based PDFs in `data/books/<subject>/<book>.pdf`. The first folder is the broad UPSC subject used by the router. For example, `data/books/polity/laxmikanth.pdf` belongs to `Polity`.

Some PDFs display readable text but use a broken font-to-Unicode mapping. PyMuPDF extracts those pages as symbols rather than words, which would produce poor embeddings. The pipeline detects this and asks for a searchable/OCRed replacement instead of indexing unreliable content.

## Ingest all books

```powershell
python main.py ingest-library --library data/books --reset
```

This creates a separate persistent ChromaDB directory for every book under `vector_dbs/`. `--reset` replaces each individual book collection, which is useful after replacing PDFs.

Optional examples:

```powershell
# Ingest only one book while developing or adding a new source
python main.py ingest-book --pdf data/books/polity/laxmikanth.pdf --subject Polity --book "Laxmikanth" --reset

# Use a different embedding model or chunk-size preference for the library
python main.py ingest-library --model all-MiniLM-L6-v2 --target-chars 900 --max-chars 1400 --reset
```

## Run a retrieval query

```powershell
python main.py query "What are the features of fundamental rights?" --top-k 3
```

The initial transparent router identifies the broad subject from question keywords and searches every database in that subject. It does **not** try to guess the exact chapter; vector search finds the relevant chapter and section. If a question is ambiguous, choose the source explicitly:

```powershell
python main.py query "Explain Article 32" --subject Polity --book "Laxmikanth" --top-k 3
```

The retrieval command returns text along with the evidence location. Its output has this form (the wording and distance depend on your PDF):

```text
#1  distance=0.2741
Subject: Polity | Book: Laxmikanth | Chapter 7: Fundamental Rights | Section: Features of Fundamental Rights | PDF page: 190 | ID: polity_laxmikanth_chunk_000421
Fundamental rights are ...
```

## Use retrieval from Python

```python
from library_retriever import LibraryRetriever

retriever = LibraryRetriever(database_root="vector_dbs")
subject, matches = retriever.search("Explain writs", top_k=3, subject="Polity")
for match in matches:
    print(match.metadata["chapter_title"], match.metadata["section"])
    print(match.text)
```

## Metadata stored per chunk

Every vector is stored together with this metadata:

```python
{
    "subject": "Polity",
    "book": "Laxmikanth",
    "chapter_number": 7,
    "chapter_title": "Fundamental Rights",
    "section": "Right to Equality",
    "chunk_id": "polity_laxmikanth_chunk_000421",
    "source_pdf_page": 198,
    "text": "... original chunk text ...",
}
```

This makes it possible for a later question-answering or evaluation layer to cite evidence from a specific chapter and section.
