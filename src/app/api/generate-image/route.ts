import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { prompt } = body;
    let imageUrl = "";

    const API_URL =
      "https://brauliopf--pentagram-text-to-image-inference-web-dev.modal.run";

    const response = await fetch(`${API_URL}/?seed=7`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ prompt }),
    });

    if (response.ok) {
      const data = await response.json();
      imageUrl = `data:${data.image.content_type};base64,${data.image.base64}`;
    } else {
      console.error("Error generating image");
    }

    return NextResponse.json({
      success: true,
      message: imageUrl,
    });
  } catch (error) {
    return NextResponse.json(
      { success: false, error: "Failed to process request" },
      { status: 500 }
    );
  }
}
