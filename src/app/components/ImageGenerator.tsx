"use client";

import { useState } from "react";

interface ImageGeneratorProps {
  generateImage: (
    text: string
  ) => Promise<{ success: boolean; imageUrl?: string; error?: string }>;
}

export default function ImageGenerator({ generateImage }: ImageGeneratorProps) {
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setImageUrl(null);
    setError(null);

    try {
      const response = await generateImage(inputText);

      if (!response.success) {
        throw new Error(response.error || "Failed to generate image");
      }

      if (response.imageUrl) {
        const img = new Image();
        const url = response.imageUrl;
        img.onload = () => {
          setImageUrl(url);
        };
        img.src = url;
      } else {
        throw new Error("No Image URL provided");
      }

      setInputText("");
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    // TODO: Update the UI here to show the images generated

    <div className="min-h-screen flex flex-col justify-between p-8">
      <main className="flex-1">
        {imageUrl && (
          <div className="w-full max-w-2x1 rounded-lg shadow-lg overflow-hidden">
            <img
              src={imageUrl}
              alt="Generated image"
              className="w-full h-auto"
            />
          </div>
        )}
      </main>

      <footer className="w-full max-w-3xl mx-auto">
        <form onSubmit={handleSubmit} className="w-full">
          <div className="flex gap-2">
            <input
              type="text"
              value={inputText}
              onChange={e => setInputText(e.target.value)}
              className="flex-1 p-3 rounded-lg bg-black/[.05] dark:bg-white/[.06] border border-black/[.08] dark:border-white/[.145] focus:outline-none focus:ring-2 focus:ring-black dark:focus:ring-white"
              placeholder="Describe the image you want to generate..."
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading}
              className="px-6 py-3 rounded-lg bg-foreground text-background hover:bg-[#383838] dark:hover:bg-[#ccc] transition-colors disabled:opacity-50"
            >
              {isLoading ? "Generating..." : "Generate"}
            </button>
          </div>
        </form>
      </footer>
    </div>
  );
}
