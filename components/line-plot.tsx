"use client";
import { useEffect, useRef, useState } from "react";
import * as Plot from '@observablehq/plot';

export default function LinePlot({ 
  data, 
  title,
  width = 400,
  height = 200,
  xLabel,
  yLabel
}: { 
  data: number[]; 
  title: string;
  width?: number;
  height?: number; 
  xLabel: string;
  yLabel: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    containerRef.current.innerHTML = '';

    const chart = Plot.plot({
      title: title,
      marks: [
        Plot.lineY(data),
      ],
      width: width,
      height: height,
      x: {
        label: xLabel
      },
      y: {
        label: yLabel
      }
    });
    
    containerRef.current.append(chart);
  }, [data, title, width, height]);

  return (
    <div ref={containerRef} />
  )

}