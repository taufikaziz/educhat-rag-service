from datetime import datetime
import os
from threading import Lock

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from app.rag_service import RAGService

app = Flask(__name__)
CORS(app)

print("\n" + "=" * 60)
print("Initializing RAG Service...")
print("=" * 60 + "\n")

_rag_service = None
_rag_lock = Lock()


def get_rag_service() -> RAGService:
    """Initialize the heavy RAG service only when first needed."""
    global _rag_service
    if _rag_service is None:
        with _rag_lock:
            if _rag_service is None:
                _rag_service = RAGService()
    return _rag_service

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "service": "EduChat RAG Service",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "health": "GET /health",
                "process": "POST /process",
                "query": "POST /query",
                "summary": "POST /summary",
            },
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "service": "RAG Service",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/process", methods=["POST"])
def process_document():
    """Process uploaded PDF."""
    print("\n" + "=" * 60)
    print("UPLOAD REQUEST RECEIVED")
    print("=" * 60)

    if "file" not in request.files:
        print("ERROR: No file in request")
        return jsonify({"success": False, "message": "No file provided"}), 400

    file = request.files["file"]
    session_id = request.form.get("session_id")

    if not session_id:
        print("ERROR: No session_id provided")
        return jsonify({"success": False, "message": "session_id required"}), 400

    print(f"Filename: {file.filename}")
    print(f"Session ID: {session_id}")

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    print(f"File saved temporarily: {filepath}")

    rag_service = get_rag_service()
    result = rag_service.process_pdf(filepath, session_id)

    try:
        os.remove(filepath)
        print("Temporary file deleted")
    except OSError:
        pass

    if result.get("success"):
        print("PROCESSING COMPLETE")
        return jsonify(result), 200
    if result.get("busy"):
        return jsonify(result), 429

    print("PROCESSING FAILED")
    return jsonify(result), 500


@app.route("/query", methods=["POST"])
def query():
    """Query the document."""
    print("\n" + "=" * 60)
    print("QUERY REQUEST RECEIVED")
    print("=" * 60)

    data = request.json or {}
    question = data.get("question")
    session_id = data.get("session_id")

    if not question or not session_id:
        print("ERROR: Missing question or session_id")
        return jsonify({"success": False, "message": "question and session_id required"}), 400

    rag_service = get_rag_service()
    result = rag_service.query(question, session_id)

    if result.get("success"):
        return jsonify(result), 200
    if result.get("busy"):
        return jsonify(result), 429
    return jsonify(result), 400


@app.route("/summary", methods=["POST"])
def summary():
    """Generate summary."""
    print("\n" + "=" * 60)
    print("SUMMARY REQUEST RECEIVED")
    print("=" * 60)

    data = request.json or {}
    session_id = data.get("session_id")

    if not session_id:
        print("ERROR: No session_id provided")
        return jsonify({"success": False, "message": "session_id required"}), 400

    rag_service = get_rag_service()
    result = rag_service.generate_summary(session_id)

    if result.get("success"):
        return jsonify(result), 200
    if result.get("busy"):
        return jsonify(result), 429
    return jsonify(result), 400


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5050"))

    print("\n" + "=" * 60)
    print("RAG SERVICE IS READY")
    print("Running on: http://localhost:{port}".format(port=port))
    print("Health check: http://localhost:{port}/health".format(port=port))
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
