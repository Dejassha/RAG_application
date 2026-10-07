import { useEffect, useState } from "react";
import axios from "axios";
import "./App.css";

const API_URL = "http://localhost:8000";

function App() {
  const [documents, setDocuments] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);

  const fetchDocuments = async () => {
    try {
      const response = await axios.get(`${API_URL}/admin/documents`);
      setDocuments(response.data.documents);
    } catch (err) {
      console.error(err);
      setError("Failed to load documents");
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleUpload = async () => {
    if (!selectedFile) {
      setError("Please select a PDF file");
      return;
    }

    if (!selectedFile.name.toLowerCase().endsWith(".pdf")) {
      setError("Only PDF files are allowed");
      return;
    }

    setUploading(true);
    setMessage("");
    setError("");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      await axios.post(`${API_URL}/admin/upload`, formData);

      setMessage("PDF uploaded successfully");
      setSelectedFile(null);

      document.getElementById("fileInput").value = "";

      await fetchDocuments();
    } catch (err) {
      console.error(err);
      setError(
        err.response?.data?.detail || "PDF upload failed"
      );
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (documentId) => {
    const confirmDelete = window.confirm(
      "Are you sure you want to delete this document?"
    );

    if (!confirmDelete) return;

    try {
      await axios.delete(
        `${API_URL}/admin/documents/${documentId}`
      );

      setMessage("Document deleted successfully");
      setError("");

      await fetchDocuments();
    } catch (err) {
      console.error(err);
      setError(
        err.response?.data?.detail || "Failed to delete document"
      );
    }
  };

  const handleSendChat = async () => {
    const question = chatInput.trim();

    if (!question || chatLoading) return;

    const updatedMessages = [
      ...chatMessages,
      { role: "user", text: question },
    ];

    setChatMessages(updatedMessages);
    setChatInput("");
    setChatLoading(true);

    try {
      const response = await axios.post(`${API_URL}/chat`, {
        question,
      });

      setChatMessages([
        ...updatedMessages,
        { role: "ai", text: response.data.answer },
      ]);
    } catch (err) {
      console.error(err);
      setChatMessages([
        ...updatedMessages,
        {
          role: "ai",
          text:
            err.response?.data?.detail ||
            "Sorry, I could not answer right now.",
        },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>ThirdVizion Admin</h1>
          <p>Manage your AI knowledge documents</p>
        </div>
      </header>

      <main className="container">

        {/* Upload Section */}
        <section className="card">
          <h2>Upload Document</h2>

          <p className="description">
            Upload a PDF document to add it to the AI knowledge base.
          </p>

          <div className="upload-area">
            <input
              id="fileInput"
              type="file"
              accept=".pdf,application/pdf"
              onChange={(e) => {
                setSelectedFile(e.target.files[0]);
                setMessage("");
                setError("");
              }}
            />

            <button
              onClick={handleUpload}
              disabled={uploading || !selectedFile}
            >
              {uploading ? "Uploading..." : "Upload PDF"}
            </button>
          </div>

          {selectedFile && (
            <p className="selected-file">
              Selected: <strong>{selectedFile.name}</strong>
            </p>
          )}

          {message && (
            <div className="success">
              {message}
            </div>
          )}

          {error && (
            <div className="error">
              {error}
            </div>
          )}
        </section>

        {/* Documents Section */}
        <section className="card">
          <div className="section-header">
            <div>
              <h2>Documents</h2>
              <p className="description">
                Documents currently available to the AI.
              </p>
            </div>

            <button
              className="refresh-btn"
              onClick={fetchDocuments}
            >
              Refresh
            </button>
          </div>

          {documents.length === 0 ? (
            <div className="empty">
              No documents found.
            </div>
          ) : (
            <div className="document-list">
              {documents.map((doc) => (
                <div
                  className="document-item"
                  key={doc.document_id}
                >
                  <div className="document-info">
                    <div className="pdf-icon">
                      PDF
                    </div>

                    <div>
                      <h3>{doc.filename}</h3>
                      <p>
                        {doc.chunks} chunks
                      </p>
                    </div>
                  </div>

                  <button
                    className="delete-btn"
                    onClick={() =>
                      handleDelete(doc.document_id)
                    }
                  >
                    Delete
                  </button>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Chatbot Section */}
        <section className="card">
          <h2>Ask AI</h2>

          <p className="description">
            Ask a question about ThirdVizion&apos;s services.
          </p>

          <div className="chat-box">
            {chatMessages.length === 0 ? (
              <div className="empty">
                No messages yet. Ask something to start.
              </div>
            ) : (
              <div className="chat-messages">
                {chatMessages.map((msg, index) => (
                  <div
                    key={index}
                    className={
                      msg.role === "user"
                        ? "chat-message user"
                        : "chat-message ai"
                    }
                  >
                    {msg.text}
                  </div>
                ))}

                {chatLoading && (
                  <div className="chat-message ai typing">
                    Typing...
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="chat-input-area">
            <input
              type="text"
              placeholder="Type your question..."
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleSendChat();
              }}
            />

            <button
              className="send-btn"
              onClick={handleSendChat}
              disabled={chatLoading || !chatInput.trim()}
            >
              {chatLoading ? "Sending..." : "Send"}
            </button>
          </div>
        </section>

      </main>
    </div>
  );
}

export default App;
