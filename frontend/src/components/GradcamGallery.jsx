import React from "react";
export default function GradcamGallery({ images }) {
  return <section className="card"><h3>Grad-CAM evidence</h3><div className="gallery">{images.map(src => <img key={src} src={`/${src.replace(/^\//, "")}`} alt="Grad-CAM visualization" />)}</div></section>;
}
