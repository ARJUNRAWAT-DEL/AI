import React from "react";

interface SearchResult {
  id: number;
  title: string;
  summary: string;
  content: string;
}

interface SearchResultsProps {
  results: SearchResult[];
}

const SearchResults: React.FC<SearchResultsProps> = ({ results }) => {
  if (!results.length) {
    return <p className="text-gray-500 text-center">No results yet.</p>;
  }

  return (
    <div className="grid gap-4">
      {results.map((result) => (
        <div
          key={result.id}
          className="bg-gray-50 border border-gray-200 rounded-lg p-4 shadow-sm text-left hover:shadow-md transition"
        >
          <h3 className="text-lg font-bold text-blue-700">{result.title}</h3>
          <p className="text-gray-600 mt-1">{result.summary}</p>
          <details className="mt-2 text-sm text-gray-500">
            <summary className="cursor-pointer text-blue-500">
              View full content
            </summary>
            <p className="mt-2">{result.content}</p>
          </details>
        </div>
      ))}
    </div>
  );
};

export default SearchResults;
