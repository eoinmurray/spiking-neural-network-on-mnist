"use client"
import * as Plot from "@observablehq/plot";
import { useEffect, useRef } from "react";

interface SpikeTrainPlotProps {
  rawData: number[][];
  width: number;
  height: number;
}

export default function SpikeTrainPlot({ rawData, width, height }: SpikeTrainPlotProps) {
  let plotData: Array<{x: number, y: number, value: number}> = [];
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
      color: {
        type: "linear",
        scheme: "greys", // built-in D3 color scheme
        // reverse: true
      },
      marks: [
        Plot.dot(plotData, {
          x: "x",
          y: "y",
          fill: "value",
          r: 1,
        }),
      ],
    });
    containerRef.current.append(plot);
    return () => plot.remove();
  }, [plotData, width, height]);

  return <div ref={containerRef} />;
}