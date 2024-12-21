"use server";

export const generateImage = async (text: string) => {
  try {
    console.log(">>>> generateImage!!!", text, process.env.API_KEY);

    // localhost is local to the container image!
    const response = await fetch("http://localhost:3000/api/generate-image", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": process.env.API_KEY || "",
      },
      body: JSON.stringify({ prompt: text }),
    });

    if (!response.ok) {
      throw new Error(`HTTP Error! status: ${response.status}`);
    }

    const data = await response.json();

    return data;
  } catch (error) {
    console.error("Server error:", error);
    return {
      sucess: false,
      error:
        error instanceof Error ? error.message : "Failed to generate image",
    };
  }
};
