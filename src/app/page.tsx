"use client";

import ImageGenerator from "./components/ImageGenerator";
import { generateImage } from "./actions/generateImage";

export default function Home() {
  return <ImageGenerator generateImage={generateImage} />;
}
