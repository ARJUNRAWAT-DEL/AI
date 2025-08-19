import React, { useState } from "react";
import SearchBar from "./components/SearchBar";
import SearchResults from "./components/SearchResults";

interface SearchResult {
  id: number;
  title: string;
  summary: string;
  content: string;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const API_URL = "http://localhost:8000";

  // ---------------- File Upload ----------------
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      alert("✅ Upload successful: " + (data.title || file.name));
    } catch (err) {
      console.error("Upload error:", err);
      alert("❌ Upload failed");
    }
  };

  // ---------------- Search ----------------
  const handleSearch = async (query: string) => {
    setLoading(true);
    setError(null);
    setResults([]);

    try {
      const res = await fetch(`${API_URL}/search?q=${encodeURIComponent(query)}`);
      const data = await res.json();

      if (Array.isArray(data.sources)) {
        setResults(data.sources);
      } else {
        setError(data.message || "No results found");
      }
    } catch (err: any) {
      console.error("Search error:", err);
      setError("Search failed: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="min-h-screen bg-gradient-to-br from-blue-100 to-purple-200 flex items-center justify-center p-6"
    >
      <div className="w-full max-w-4xl bg-white/95 shadow-2xl rounded-2xl p-8 text-center">
        {/* Hero Section */}
        <div className="mb-10">
          <img
            src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
            alt="AI Search"
            className="mx-auto w-24 h-24 mb-4 drop-shadow-lg rounded-full"
          />
          <h1 className="text-4xl font-extrabold text-blue-700">
            AI Document Search
          </h1>
          <p className="text-gray-600 mt-2 text-lg">
            Upload documents and find answers instantly with AI-powered search
          </p>
        </div>

        {/* File Upload Card */}
        <div className="bg-white shadow-md rounded-2xl p-6 mb-8">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">
            Upload Document
          </h2>
          <div className="flex flex-col sm:flex-row items-center gap-4 justify-center">
            <input
              type="file"
              onChange={handleFileChange}
              className="block w-full text-sm text-gray-600 
                         file:mr-4 file:py-2 file:px-4 
                         file:rounded-lg file:border-0 file:text-sm file:font-semibold 
                         file:bg-blue-100 file:text-blue-700 hover:file:bg-blue-200"
            />
            <button
              onClick={handleUpload}
              className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg shadow-md transition"
            >
              Upload
            </button>
          </div>
        </div>

        {/* Search Card */}
        <div className="bg-white shadow-md rounded-2xl p-6 mb-8">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">
            Search Documents
          </h2>
          <div className="flex justify-center">
            <SearchBar onSearch={handleSearch} />
          </div>
        </div>

        {/* Results */}
        <div className="bg-white shadow-md rounded-2xl p-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Results</h2>
          {loading && <p className="text-gray-500 text-center">🔍 Searching...</p>}
          {error && <p className="text-red-600 text-center">{error}</p>}
          <SearchResults results={results} />
        </div>
      </div>
    </div>
  );
}

export default App;
