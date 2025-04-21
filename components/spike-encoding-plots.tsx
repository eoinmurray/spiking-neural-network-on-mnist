"use client";
import { useEffect, useRef, useState } from "react";
import * as Plot from '@observablehq/plot';

export default function SpikeEncodingPlots({ 
  data, 
  title,
  width = 400,
  height = 200,
}: { 
  data: any; 
  title: string;
  width?: number;
  height?: number; 
}) {
  const chart1Ref = useRef<HTMLDivElement>(null);
  const chart2Ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!chart1Ref.current) return;
    if (!chart2Ref.current) return;

    chart1Ref.current.innerHTML = '';
    chart2Ref.current.innerHTML = '';

    const zippedData = data.neuron_ids.map((id: number, index: number) => ({
      neuron_id: id,
      spike_time: data.spike_times[index],
    }));

    const chart1 = Plot.plot({
      title,
      marks: [
        Plot.dot(zippedData, {
          x: "spike_time",
          y: "neuron_id",
          r: 0.5,
        }),
      ],
      width: width,
      height: height,
    });

    const tidyImageData = data.image.flatMap((row: number[], y: number) =>
      row.map((value: number, x: number) => ({
        x,
        y,
        value,
      }))
    );

    const chart2 = Plot.plot({
      title: "Original Image",
      marks: [
        Plot.cell(tidyImageData, {
          x: "x",
          y: "y",
          fill: d => d.value,
        }),
      ],
      width: width,
      height: height,
    });
    
    chart1Ref.current.append(chart1);
    chart2Ref.current.append(chart2);

  }, [data, title, width, height]);

  return (
    <div className="grid grid-cols-2 gap-4">
      <div ref={chart2Ref} />
      <div ref={chart1Ref} />
    </div>
  )
}