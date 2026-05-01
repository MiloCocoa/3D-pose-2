import React, { useMemo } from 'react';
import { Line, Sphere } from '@react-three/drei';
import * as THREE from 'three';
import { SKELETON_EDGES } from '../constants';

interface Skeleton3DProps {
  pose: number[][]; // (33, 3)
  mistakes?: string[];
  jointHeatmap?: number[]; // (36,) - Severity scores from 0.0 to 1.0
}

const Skeleton3D: React.FC<Skeleton3DProps> = ({ pose, mistakes = [], jointHeatmap = [] }) => {
  const points = useMemo(() => {
    return pose.map(p => {
        if (!p || p.length < 3) return new THREE.Vector3(0, 0, 0);
        return new THREE.Vector3(p[0] || 0, -(p[1] || 0), -(p[2] || 0));
    });
  }, [pose]);

  // Helper to interpolate color based on severity
  // 0.0 -> Green (#10b981)
  // 1.0 -> Red (#ef4444)
  const getHeatmapColor = (severity: number, isVirtual: boolean) => {
    if (severity <= 0) return isVirtual ? "#fbbf24" : "#10b981";
    
    const color = new THREE.Color("#10b981"); // Green
    const target = new THREE.Color("#ef4444"); // Red
    color.lerp(target, severity);
    return "#" + color.getHexString();
  };

  return (
    <group>
      {/* Joints */}
      {points.map((p, i) => {
        const severity = jointHeatmap[i] || 0;
        const isVirtual = i >= 33;
        
        return (
          <Sphere key={i} position={p} args={[severity > 0 ? 0.03 : 0.02, 16, 16]}>
            <meshBasicMaterial 
              color={getHeatmapColor(severity, isVirtual)} 
              opacity={severity > 0 ? 1.0 : 0.8}
              transparent
            />
          </Sphere>
        );
      })}

      {/* Edges */}
      {SKELETON_EDGES.map(([start, end], i) => {
        if (!points[start] || !points[end]) return null;
        
        const s_sev = jointHeatmap[start] || 0;
        const e_sev = jointHeatmap[end] || 0;
        const edge_sev = (s_sev + e_sev) / 2;
        
        const isVirtual = start >= 33 || end >= 33;
        
        return (
            <Line
                key={i}
                points={[points[start], points[end]]}
                color={getHeatmapColor(edge_sev, isVirtual)}
                lineWidth={edge_sev > 0 ? 3 : (isVirtual ? 1 : 2)}
                transparent
                opacity={edge_sev > 0 ? 1.0 : 0.6}
            />
        );
      })}
    </group>
  );
};

export default Skeleton3D;
