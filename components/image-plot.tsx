"use client"
import * as Plot from "@observablehq/plot";
import { useEffect, useRef } from "react";

interface ImagePlotProps {
  rawData: number[][];
  width: number;
  height: number;
}

export default function ImagePlot({ rawData, width, height }: ImagePlotProps) {
  const plotData: Array<{x: number, y: number, value: number}> = [];
  for (let y = 0; y < rawData.length; y++) {
    for (let x = 0; x < rawData[y].length; x++) {
      plotData.push({ x, y, value: rawData[y][x] });
    }
  }

  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (plotData === undefined || !containerRef.current) return;
    const plot = Plot.plot({      
      width, 
      height,
      marks: [
        Plot.rect(plotData, {
          x: "x",
          y: "y",
          fill: "value",
        }),
      ],
      x: { axis: null },
      y: { axis: null },
    });
    containerRef.current.append(plot);
    return () => plot.remove();
  }, [plotData, width, height]);

  return <div ref={containerRef} />;
}