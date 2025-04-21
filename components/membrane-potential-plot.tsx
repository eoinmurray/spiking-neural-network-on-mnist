"use client"
import * as Plot from "@observablehq/plot";
import { useEffect, useRef } from "react";

export default function MembranePotentialPlot({ 
  beforeResetData,
  afterResetData,
  threshold,
  width, 
  height
}: {
  beforeResetData: number[][];
  afterResetData: number[][];
  threshold: number;
  width: number;
  height: number;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  const offsetMultiplier = 2.0

  useEffect(() => {
    if (beforeResetData === undefined || !containerRef.current) return;
    const plot = Plot.plot({      
      width, 
      height,
      marks: [
        Plot.ruleX([0]),
        Plot.ruleY([0]),
        beforeResetData.map((lineData, index) => {
          return Plot.lineY(lineData.map(d => d + index * offsetMultiplier), { stroke: "black", strokeWidth: 0.5 })
        }),
        beforeResetData.map((lineData, index) => {
          return Plot.ruleY([threshold + index * offsetMultiplier], { stroke: "red", strokeWidth: 0.2, strokeDasharray: "4 2" })
        })
      ].flat(),
    });
    containerRef.current.append(plot);
    return () => plot.remove();
  }, [beforeResetData, afterResetData, threshold, width, height, offsetMultiplier]);

  return <div ref={containerRef} />;
}