import { useState } from "react";
import "./App.css";

function App() {
  // ============================================================
  // PDF UPLOAD STATE
  // ============================================================

  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  const [uploadError, setUploadError] = useState("");


  // ============================================================
  // QUESTION / ANSWER STATE
  // ============================================================

  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");


  // ============================================================
  // HANDLE FILE SELECTION
  // ============================================================

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];

    setFile(selectedFile);

    setUploadMessage("");
    setUploadError("");
  };


  // ============================================================
  // UPLOAD PDF
  // ============================================================

  const uploadPDF = async () => {
    if (!file) {
      setUploadError("Please select a PDF first.");
      return;
    }


    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Only PDF files are supported.");
      return;
    }


    setUploading(true);
    setUploadMessage("");
    setUploadError("");


    try {
      const formData = new FormData();

      formData.append("file", file);


      const response = await fetch(
        "http://127.0.0.1:8000/upload",
        {
          method: "POST",
          body: formData,
        }
      );


      if (!response.ok) {
        const errorData = await response.json();

        throw new Error(
          errorData.detail ||
          "PDF upload failed."
        );
      }


      const data = await response.json();


      setUploadMessage(
        `${data.filename} uploaded successfully. ` +
        `${data.pages} pages and ${data.chunks} chunks indexed.`
      );

    } catch (error) {

      console.error(error);

      setUploadError(
        error.message ||
        "Could not upload the PDF."
      );

    } finally {

      setUploading(false);

    }
  };


  // ============================================================
  // ASK QUESTION
  // ============================================================

  const askQuestion = async () => {

    if (!question.trim()) {
      return;
    }


    setLoading(true);

    setAnswer("");

    setSources([]);

    setError("");


    try {

      const response = await fetch(
        "http://127.0.0.1:8000/ask",
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify({
            question: question,
          }),
        }
      );


      if (!response.ok) {

        const errorData = await response.json();

        throw new Error(
          errorData.detail ||
          "Failed to get response from server."
        );
      }


      const data = await response.json();


      setAnswer(data.answer);

      setSources(data.sources || []);

    } catch (error) {

      console.error(error);

      setError(
        error.message ||
        "Could not connect to the AI backend."
      );

    } finally {

      setLoading(false);

    }
  };


  // ============================================================
  // ENTER KEY
  // ============================================================

  const handleKeyDown = (event) => {

    if (event.key === "Enter") {

      askQuestion();

    }

  };


  // ============================================================
  // UI
  // ============================================================

  return (

    <div className="app">

      {/* ======================================================
          HEADER
      ====================================================== */}

      <header className="header">

        <h1>
          AI Knowledge Assistant
        </h1>

        <p>
          Upload documents and ask questions using RAG
        </p>

      </header>


      <main className="container">


        {/* ====================================================
            PDF UPLOAD
        ==================================================== */}

        <section className="upload-card">

          <h2>
            📄 Upload PDF
          </h2>

          <p>
            Upload a document to build the knowledge base.
          </p>


          <div className="upload-section">

            <input
              type="file"
              accept=".pdf,application/pdf"
              onChange={handleFileChange}
            />


            <button
              onClick={uploadPDF}
              disabled={uploading}
            >

              {uploading
                ? "Uploading..."
                : "Upload PDF"}

            </button>

          </div>


          {file && (
            <p className="selected-file">
              Selected: {file.name}
            </p>
          )}


          {uploadMessage && (
            <div className="success">
              {uploadMessage}
            </div>
          )}


          {uploadError && (
            <div className="error">
              {uploadError}
            </div>
          )}

        </section>


        {/* ====================================================
            QUESTION SECTION
        ==================================================== */}

        <section className="question-card">

          <h2>
            💬 Ask a Question
          </h2>


          <div className="input-section">

            <input
              type="text"
              placeholder="Ask something about your PDF..."
              value={question}
              onChange={(event) =>
                setQuestion(event.target.value)
              }
              onKeyDown={handleKeyDown}
            />


            <button
              onClick={askQuestion}
              disabled={loading}
            >

              {loading
                ? "Thinking..."
                : "Ask"}

            </button>

          </div>

        </section>


        {/* ====================================================
            ERROR
        ==================================================== */}

        {error && (

          <div className="error">
            {error}
          </div>

        )}


        {/* ====================================================
            ANSWER
        ==================================================== */}

        {answer && (

          <section className="answer-card">

            <h2>
              🤖 AI Answer
            </h2>

            <p>
              {answer}
            </p>

          </section>

        )}


        {/* ====================================================
            SOURCES
        ==================================================== */}

        {sources.length > 0 && (

          <section className="sources-card">

            <h2>
              📚 Sources
            </h2>


            {sources.map(
              (source, index) => (

                <div
                  className="source"
                  key={index}
                >

                  <strong>
                    📄 {source.document}
                  </strong>


                  <span>
                    Page {source.page}
                  </span>

                </div>

              )
            )}

          </section>

        )}

      </main>

    </div>

  );
}

export default App;