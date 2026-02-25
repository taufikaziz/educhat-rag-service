import os
import re
import warnings
from threading import Lock
from typing import Dict, List, Optional

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
os.environ.setdefault("CHROMA_TELEMETRY", "FALSE")

from pypdf.errors import PdfReadWarning
from dotenv import load_dotenv
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq

load_dotenv()
# Suppress noisy non-fatal PDF structure warnings from certain generators.
warnings.filterwarnings(
    "ignore",
    message=r"Multiple definitions in dictionary.*",
    category=PdfReadWarning,
)


class RAGService:
    def __init__(self):
        print("Initializing RAG Service...")

        self.persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        os.makedirs(self.persist_directory, exist_ok=True)
        print(f"ChromaDB directory: {self.persist_directory}")
        self.chroma_settings = Settings(anonymized_telemetry=False)

        print("Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("Embeddings ready")

        print("Connecting to Groq LLM...")
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
        print("LLM ready")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=150,
        )
        self.default_query_mode = os.getenv("RAG_QUERY_MODE", "fast").lower()
        self.max_pdf_pages = int(os.getenv("MAX_PDF_PAGES", "80"))
        self.operation_lock = Lock()

        print("RAG Service initialized successfully\n")

    @staticmethod
    def _collection_count(vectorstore: Chroma) -> int:
        """Safely read collection size for adaptive retrieval params."""
        try:
            collection = getattr(vectorstore, "_collection", None)
            if collection is None:
                return 0
            count = collection.count()
            return int(count) if count is not None else 0
        except Exception:
            return 0

    def _expand_queries(self, question: str) -> List[str]:
        """Generate focused retrieval queries from one user question."""
        prompt = f"""Kamu adalah query rewriter untuk sistem RAG.

Buat 4 variasi query pencarian dalam Bahasa Indonesia berdasarkan pertanyaan pengguna.
Aturan:
- Fokus ke keyword konsep inti, istilah teknis, dan sinonim penting
- Jangan menjawab pertanyaan
- Setiap baris hanya 1 query
- Maksimal 12 kata per query
- Jangan gunakan bullet atau nomor

Pertanyaan pengguna: {question}

Output:"""

        try:
            response = self.llm.invoke(prompt)
            lines = [line.strip("- *\t ") for line in response.content.splitlines()]
            queries = [line for line in lines if line]
        except Exception:
            queries = []

        # Always include the original user query.
        merged = [question]
        for query in queries:
            if query.lower() not in [q.lower() for q in merged]:
                merged.append(query)
            if len(merged) >= 5:
                break

        return merged

    def _retrieve_context(self, vectorstore: Chroma, question: str, mode: str = "fast") -> List[Document]:
        """Retrieve and deduplicate relevant chunks using multi-query + MMR."""
        use_deep_mode = mode == "deep"
        queries = self._expand_queries(question) if use_deep_mode else [question]
        all_docs: List[Document] = []
        requested_page = self._extract_requested_page(question)
        collection_size = self._collection_count(vectorstore)
        default_fetch_k = 20 if use_deep_mode else 10
        default_k = 4 if use_deep_mode else 3
        mmr_fetch_k = min(default_fetch_k, collection_size) if collection_size > 0 else default_fetch_k

        if requested_page is not None:
            print(f"Detected explicit page request: {requested_page}")
            page_docs = self._retrieve_for_page(vectorstore, question, requested_page)
            all_docs.extend(page_docs)

        for query in queries:
            docs = vectorstore.max_marginal_relevance_search(query, k=default_k, fetch_k=mmr_fetch_k)
            all_docs.extend(docs)

        seen = set()
        unique_docs: List[Document] = []
        for doc in all_docs:
            key = (doc.page_content[:200], doc.metadata.get("page", -1))
            if key in seen:
                continue
            seen.add(key)
            unique_docs.append(doc)

        return unique_docs[:8]

    @staticmethod
    def _extract_requested_page(question: str) -> Optional[int]:
        """Extract requested page/slide number from user question."""
        lower_question = question.lower()
        patterns = [
            r"\b(?:slide|halaman|page)\s*(?:ke|ke-)?\s*(\d+)\b",
            r"\bke\s*[-]?\s*(\d+)\s*(?:slide|halaman|page)\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, lower_question)
            if match:
                number = int(match.group(1))
                if number > 0:
                    return number
        return None

    def _retrieve_for_page(self, vectorstore: Chroma, question: str, page_number: int) -> List[Document]:
        """Retrieve chunks filtered by a specific page (1-based page_number)."""
        page_index = page_number - 1
        collection_size = self._collection_count(vectorstore)
        fetch_k = min(30, collection_size) if collection_size > 0 else 30

        try:
            docs = vectorstore.max_marginal_relevance_search(
                question,
                k=6,
                fetch_k=fetch_k,
                filter={"page": page_index},
            )
            if docs:
                return docs
        except TypeError:
            docs = vectorstore.similarity_search(question, k=6, filter={"page": page_index})
            if docs:
                return docs
        except Exception:
            pass

        docs: List[Document] = []
        try:
            raw = vectorstore.get(
                where={"page": page_index},
                include=["documents", "metadatas"],
                limit=8,
            )
            documents = raw.get("documents") or []
            metadatas = raw.get("metadatas") or []

            for idx, text in enumerate(documents):
                if not text:
                    continue
                metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {"page": page_index}
                docs.append(Document(page_content=text, metadata=metadata))
        except Exception:
            return []

        return docs

    @staticmethod
    def _format_context(docs: List[Document]) -> str:
        parts = []
        for idx, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page")
            page_label = f"Halaman {page + 1}" if isinstance(page, int) else "Halaman tidak diketahui"
            parts.append(f"[Sumber {idx} | {page_label}]\n{doc.page_content}")
        return "\n\n".join(parts)

    def process_pdf(self, file_path: str, session_id: str) -> Dict:
        """Process PDF and store in ChromaDB."""
        if not self.operation_lock.acquire(timeout=3):
            return {
                "success": False,
                "busy": True,
                "message": "Server sedang memproses request lain. Coba lagi beberapa detik.",
            }
        try:
            try:
                print(f"\n{'=' * 60}")
                print(f"Processing PDF: {os.path.basename(file_path)}")
                print(f"Session ID: {session_id}")
                print(f"{'=' * 60}\n")

                print("Loading PDF...")
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                print(f"Loaded {len(documents)} pages")
                if len(documents) > self.max_pdf_pages:
                    return {
                        "success": False,
                        "message": (
                            f"Jumlah halaman terlalu besar ({len(documents)}). "
                            f"Maksimal {self.max_pdf_pages} halaman per upload agar server stabil."
                        ),
                        "num_chunks": 0,
                        "num_pages": len(documents),
                    }
                text_pages = sum(1 for doc in documents if doc.page_content and doc.page_content.strip())
                print(f"Pages with extractable text: {text_pages}")

                if text_pages == 0:
                    return {
                        "success": False,
                        "message": (
                            "PDF tidak memiliki teks yang bisa diekstrak (kemungkinan hasil scan/gambar). "
                            "Silakan gunakan PDF berbasis teks atau jalankan OCR terlebih dahulu."
                        ),
                        "num_chunks": 0,
                        "num_pages": len(documents),
                    }

                print("Splitting into chunks...")
                chunks = self.text_splitter.split_documents(documents)
                chunks = [chunk for chunk in chunks if chunk.page_content and chunk.page_content.strip()]
                print(f"Created {len(chunks)} chunks")

                if not chunks:
                    return {
                        "success": False,
                        "message": (
                            "Teks pada PDF terlalu minim atau kosong, sehingga tidak ada chunk yang bisa diproses."
                        ),
                        "num_chunks": 0,
                        "num_pages": len(documents),
                    }

                print("Adding metadata...")
                for chunk in chunks:
                    chunk.metadata["session_id"] = session_id

                print("Storing in ChromaDB...")
                Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                    collection_name=f"session_{session_id.replace('-', '_')}",
                    client_settings=self.chroma_settings,
                )

                print(f"\nSUCCESS: Processed and stored {len(chunks)} chunks")
                print(f"{'=' * 60}\n")

                return {
                    "success": True,
                    "message": f"Successfully processed {len(chunks)} chunks",
                    "num_chunks": len(chunks),
                    "num_pages": len(documents),
                }

            except Exception as e:
                print(f"\nERROR: {str(e)}\n")
                import traceback

                traceback.print_exc()
                return {
                    "success": False,
                    "message": str(e),
                }
        finally:
            self.operation_lock.release()

    def query(self, question: str, session_id: str, mode: str = "fast") -> Dict:
        """Query the RAG system."""
        if not self.operation_lock.acquire(timeout=3):
            return {
                "success": False,
                "busy": True,
                "message": "Server sedang memproses request lain. Coba lagi beberapa detik.",
            }
        try:
            try:
                selected_mode = mode.lower().strip() if mode else self.default_query_mode
                if selected_mode not in {"fast", "deep"}:
                    selected_mode = self.default_query_mode if self.default_query_mode in {"fast", "deep"} else "fast"

                print(f"\n{'=' * 60}")
                print("Query received")
                print(f"Question: {question}")
                print(f"Session ID: {session_id}")
                print(f"Mode: {selected_mode}")
                print(f"{'=' * 60}\n")

                collection_name = f"session_{session_id.replace('-', '_')}"
                print(f"Loading collection: {collection_name}")

                vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=collection_name,
                    client_settings=self.chroma_settings,
                )

                print("Searching relevant chunks...")
                docs = self._retrieve_context(vectorstore, question, mode=selected_mode)
                requested_page = self._extract_requested_page(question)

                if not docs:
                    print("No relevant context found")
                    return {
                        "success": False,
                        "message": "No context found. Please upload a document first.",
                    }

                if requested_page is not None:
                    has_requested_page = any(doc.metadata.get("page") == requested_page - 1 for doc in docs)
                    if not has_requested_page:
                        print("Requested page not found in retrieved context")
                        return {
                            "success": False,
                            "message": f"Halaman/slide {requested_page} tidak ditemukan pada materi yang tersimpan.",
                        }

                print(f"Found {len(docs)} relevant chunks")
                context = self._format_context(docs)

                prompt = f"""Kamu adalah asisten pembelajaran untuk mahasiswa.

Tugasmu: jawab pertanyaan hanya berdasarkan konteks materi.
Jika informasi tidak cukup, katakan: "Maaf, informasi tersebut tidak ada dalam materi yang diupload."

Aturan jawaban:
- Gunakan Bahasa Indonesia yang jelas dan mudah dipahami
- Fokus pada fakta dari konteks
- Untuk pertanyaan kompleks, jelaskan langkah demi langkah
- Jika pertanyaan menyebut slide/halaman tertentu, prioritaskan sumber dari halaman tersebut
- Sertakan bagian "Referensi" di akhir berupa daftar sumber yang dipakai, misalnya: Sumber 1 (Halaman 3)
- Jangan mengarang informasi di luar konteks

Konteks materi:
{context}

Pertanyaan:
{question}

Jawaban:"""

                print("Generating answer...")
                response = self.llm.invoke(prompt)

                print("Answer generated")
                print(f"{'=' * 60}\n")

                sources = []
                for idx, doc in enumerate(docs, start=1):
                    page = doc.metadata.get("page")
                    page_label = page + 1 if isinstance(page, int) else None
                    sources.append(
                        {
                            "source_id": idx,
                            "page": page_label,
                            "preview": doc.page_content[:180],
                        }
                    )

                return {
                    "success": True,
                    "mode": selected_mode,
                    "answer": response.content,
                    "sources": sources,
                }

            except Exception as e:
                print(f"\nERROR: {str(e)}\n")
                import traceback

                traceback.print_exc()
                return {
                    "success": False,
                    "message": str(e),
                }
        finally:
            self.operation_lock.release()

    def generate_summary(self, session_id: str) -> Dict:
        """Generate summary of document."""
        if not self.operation_lock.acquire(timeout=3):
            return {
                "success": False,
                "busy": True,
                "message": "Server sedang memproses request lain. Coba lagi beberapa detik.",
            }
        try:
            try:
                print(f"\n{'=' * 60}")
                print("Generating summary")
                print(f"Session ID: {session_id}")
                print(f"{'=' * 60}\n")

                collection_name = f"session_{session_id.replace('-', '_')}"
                print(f"Loading collection: {collection_name}")

                vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=collection_name,
                    client_settings=self.chroma_settings,
                )

                print("Retrieving document chunks...")
                collection_size = self._collection_count(vectorstore)
                summary_fetch_k = min(30, collection_size) if collection_size > 0 else 30
                docs = vectorstore.max_marginal_relevance_search(
                    "ringkasan materi pembelajaran konsep utama definisi contoh",
                    k=12,
                    fetch_k=summary_fetch_k,
                )

                if not docs:
                    print("No document found")
                    return {
                        "success": False,
                        "message": "No document found. Please upload first.",
                    }

                print(f"Retrieved {len(docs)} chunks")
                context = "\n\n".join([doc.page_content for doc in docs])

                prompt = f"""Buat ringkasan materi berikut dalam bahasa Indonesia yang mudah dipahami.

Format:
1. Topik Utama
2. Poin Penting (5-7 poin)
3. Istilah Kunci + definisi singkat
4. Kesimpulan

Materi:
{context[:4500]}

Ringkasan:"""

                print("Generating summary...")
                response = self.llm.invoke(prompt)

                print("Summary generated")
                print(f"{'=' * 60}\n")

                return {
                    "success": True,
                    "summary": response.content,
                }

            except Exception as e:
                print(f"\nERROR: {str(e)}\n")
                import traceback

                traceback.print_exc()
                return {
                    "success": False,
                    "message": str(e),
                }
        finally:
            self.operation_lock.release()
