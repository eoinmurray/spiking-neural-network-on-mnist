"use client";
import { useEffect, useRef, useState } from "react";
import * as Plot from '@observablehq/plot';

export default function StackedHistogram({ 
  rawData,
  title,
  xLabel = "Weight Value",
  yLabel = "Counts",
  width = 400,
  height = 200,
}: { 
  rawData: number[],
  title?: string;
  width?: number;
  height?: number; 
  xLabel?: string;
  yLabel?: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    containerRef.current.innerHTML = '';

    const chart = Plot.plot({
      title: title,
      marks: [
        Plot.rectY(rawData, Plot.binX({y: "count", thresholds: 50})),
        Plot.ruleY([0])
      ],
      width: width,
      height: height,
      x: {
        label: xLabel,
      },
      y: {
        label: yLabel,
      }
    });
    
    containerRef.current.append(chart);
  }, [rawData, title, width, height]);

  return (
    <div ref={containerRef} />
  )

}