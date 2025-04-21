"use client";
import { useEffect, useRef, useState } from "react";
import * as Plot from '@observablehq/plot';

export default function Histogram({ 
  counts, 
  bins,
  title,
  width = 400,
  height = 200,
  xLabel = "Value",
  yLabel = "Frequency"
}: { 
  counts: number[]; 
  bins: number[];
  title: string;
  width?: number;
  height?: number; 
  xLabel?: string;
  yLabel?: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  const data = bins.map((bin, index) => ({
    bin,
    count: counts[index] || 0,
  }));

  useEffect(() => {
    if (!containerRef.current) return;
    containerRef.current.innerHTML = '';

    const chart = Plot.plot({
      title: title,
      marks: [
        Plot.lineY(data, { x: d => d.bin, y: d => d.count}),
      ],
      width: width,
      height: height,
      x: {
        label: xLabel,
        ticks: 10,
      },
      y: {
        label: yLabel,
        domain: [0, Math.max(...counts) * 1.1],
      }
    });
    
    containerRef.current.append(chart);
  }, [counts, bins, title, width, height]);

  return (
    <div ref={containerRef} />
  )

}