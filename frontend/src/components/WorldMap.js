import { useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { feature } from 'topojson-client';

const WorldMap = ({ data, topology }) => {
  const svgRef = useRef(null);

  useEffect(() => {
    const createMap = async () => {
      // Clear any existing SVG content
      d3.select(svgRef.current).selectAll("*").remove();
      
      // Set dimensions
      const width = 800;
      const height = 450;
      
      // Create SVG
      const svg = d3.select(svgRef.current)
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [0, 0, width, height])
        .attr("style", "max-width: 100%; height: auto;");
        
      // Create a projection
      const projection = d3.geoNaturalEarth1()
        .scale(width / 2 / Math.PI)
        .translate([width / 2, height / 2]);
        
      // Create a path generator
      const path = d3.geoPath()
        .projection(projection);
      
      // Convert TopoJSON to GeoJSON
      const countries = feature(topology, topology.objects.countries);
      
      // Draw the map
      svg.append("g")
        .selectAll("path")
        .data(countries.features)
        .join("path")
        .attr("fill", "#ccc")  // Grey background
        .attr("d", path)
        .attr("stroke", "#fff")
        .attr("stroke-width", 0.5);
    };
    
    createMap();
  }, [data, topology]);

  return (
    <div className="world-map-container">
      <h2>World Language Distribution</h2>
      <svg ref={svgRef}></svg>
    </div>
  );
};

export default WorldMap; 