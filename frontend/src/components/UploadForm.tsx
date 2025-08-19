import { useState } from "react";

export default function UploadForm() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState("");
  const [uploadedFile, setUploadedFile] = useState<{ name: string; size: number } | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    setStatus("Uploading...");
    setUploadedFile(null);

    try {
     const res = await fetch("http://localhost:8000/upload-file", {
    method: "POST",
    body: formData,
  });


      if (res.ok) {
        const data = await res.json();
        setStatus("Upload successful!");
        setUploadedFile({ name: data.filename, size: data.size });
      } else {
        setStatus("Upload failed.");
      }
    } catch (err) {
      console.error(err);
      setStatus("Error uploading file.");
    }
  };

  return (
    <div className="p-4 border rounded-md mb-8">
      <input type="file" onChange={handleFileChange} className="mb-4" />
      <button
        onClick={handleUpload}
        className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
      >
        Upload
      </button>
      <p className="mt-2 text-sm text-gray-600">{status}</p>
      {uploadedFile && (
        <div className="mt-2 text-sm text-green-700">
          <p>Uploaded file: {uploadedFile.name}</p>
          <p>Size: {(uploadedFile.size / 1024).toFixed(2)} KB</p>
        </div>
      )}
    </div>
  );
}
