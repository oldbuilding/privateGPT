import os
import glob
from typing import List
from dotenv import load_dotenv

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    PythonLoader,
    JSONLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS


load_dotenv()


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    # ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    # ".docx": (UnstructuredWordDocumentLoader, {}),
    # ".enex": (EverNoteLoader, {}),
    # ".eml": (UnstructuredEmailLoader, {}),
    # ".epub": (UnstructuredEPubLoader, {}),
    # ".html": (UnstructuredHTMLLoader, {}),
    # ".md": (UnstructuredMarkdownLoader, {}),
    # ".odt": (UnstructuredODTLoader, {}),
    # ".pdf": (PDFMinerLoader, {}),
    # ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".h": (TextLoader, {"encoding": "utf8"}),
    ".c": (TextLoader, {"encoding": "utf8"}),
    ".py": (PythonLoader, {}),
    # ".mk": (TextLoader, {"encoding": "utf8"}),
    # ".json": (JSONLoader, {"jq_schema": ".[]", "text_content": False}),
    # Add more mappings for other file extensions and loaders as needed
}


load_dotenv()


def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext not in LOADER_MAPPING:
        raise ValueError(f"Unsupported file extension '{ext}'")

    loader_class, loader_args = LOADER_MAPPING[ext]
    try:
        print(f"-> {file_path}")
        loader = loader_class(file_path, **loader_args)
        document = loader.load()
        if document:
            return document[0]
        else:
            print(f"Warning: No document was loaded from {file_path}")
            return None
    except Exception:
        print(f"Failed to load document '{file_path}'")


def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from source documents directory
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    documents = [load_single_document(file_path) for file_path in all_files]
    return [doc for doc in documents if doc is not None]


def main():
    # Load environment variables
    persist_directory = os.environ.get("PERSIST_DIRECTORY")
    source_directory = os.environ.get(
        "SOURCE_DIRECTORY", "source_documents/qmk"
    )  # source_documents')
    embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")

    # Load documents and split in chunks
    print(f"Loading documents from {source_directory}")
    chunk_size = 100
    chunk_overlap = 10
    documents = load_documents(source_directory)
    print("completed load_documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents from {source_directory}")
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} characters each)")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    # Create and store locally vectorstore
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=persist_directory,
        client_settings=CHROMA_SETTINGS,
    )
    db.persist()
    db = None


if __name__ == "__main__":
    main()
