import { NextResponse } from "next/server";
import { put } from "@vercel/blob";
import crypto from "crypto";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { prompt } = body;

    const url = new URL(
      "https://brauliopf--text-2-image-demo-model-generate-dev.modal.run/"
    );
    url.searchParams.set("prompt", prompt);

    const response = await fetch(url, {
      method: "GET",
      headers: {
        "X-API-Key": process.env.API_KEY || "",
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Error generating image", errorText);
      throw new Error(
        `HTTP Error! status: ${response.status}. message: ${errorText}`
      );
    }

    // request was successful.  handle image blob
    const imageBuffer = await response.arrayBuffer();
    const filename = `${crypto.randomUUID()}.jpg`;

    const blob = await put(filename, imageBuffer, {
      access: "public",
      contentType: "image/jpeg",
    });

    return NextResponse.json({
      success: true,
      imageUrl: blob.url,
    });
  } catch (error) {
    return NextResponse.json(
      { success: false, error: "Failed to process request" },
      { status: 500 }
    );
  }
}
