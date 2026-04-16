import React, { Suspense, useRef, useCallback, useEffect } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Stage, useGLTF, Environment } from '@react-three/drei';
import * as THREE from 'three';

// Error boundary to catch Three.js / Canvas crashes
class ViewerErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="w-full h-full bg-slate-900 flex items-center justify-center">
          <div className="text-center space-y-4 p-8">
            <span className="text-5xl block text-red-400">&#x26A0;</span>
            <p className="text-lg font-medium text-slate-300">Model failed to load</p>
            <p className="text-sm text-slate-500 max-w-xs">
              {this.state.error?.message || 'An unexpected rendering error occurred.'}
            </p>
            <button
              onClick={this.handleRetry}
              className="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold rounded-lg transition-colors"
            >
              Click to retry
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

// Matches mesh names produced by the glTF exporter: "wall_0", "column_2", etc.
const ELEMENT_NAME_RE = /^(wall|column|door|window|floor|ceiling)_(\d+)$/;

/**
 * Walk from the clicked Three.js object upward through ancestors to find
 * the first node whose name matches an element type+index pattern.
 */
function resolveElementMesh(object) {
  let current = object;
  while (current) {
    if (ELEMENT_NAME_RE.test(current.name)) return current;
    current = current.parent;
  }
  return null;
}

const SelectableModel = ({ url, onElementSelect, selectedMesh }) => {
  const { scene } = useGLTF(url);
  const { camera } = useThree();
  const prevRef = useRef(null);

  // Zoom-to-fit: frame the entire model after it loads
  useEffect(() => {
    if (!scene) return;
    const box = new THREE.Box3().setFromObject(scene);
    if (box.isEmpty()) return;

    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = camera.fov * (Math.PI / 180);
    // Distance to fit object in view, with 1.5x padding
    const distance = (maxDim / (2 * Math.tan(fov / 2))) * 1.5;

    camera.position.set(
      center.x + distance * 0.5,
      center.y + distance * 0.6,
      center.z + distance * 0.7
    );
    camera.lookAt(center);
    camera.updateProjectionMatrix();
  }, [scene, camera]);

  const applyHighlight = (mesh) => {
    if (!mesh?.material) return;
    const mat = mesh.material;
    // Save original state for later restoration
    mesh.userData._origColor    = mat.color?.clone?.() ?? null;
    mesh.userData._origEmissive = mat.emissive?.clone?.() ?? null;
    mesh.userData._origEI       = mat.emissiveIntensity ?? 0;
    // Apply orange highlight
    if (mat.emissive !== undefined) {
      mat.emissive.set(0xffaa00);
      mat.emissiveIntensity = 0.7;
    } else if (mat.color) {
      mat.color.set(0xffdd55);
    }
    mat.needsUpdate = true;
  };

  const clearHighlight = (mesh) => {
    if (!mesh?.material) return;
    const mat = mesh.material;
    if (mesh.userData._origEmissive !== null && mat.emissive !== undefined) {
      mat.emissive.copy(mesh.userData._origEmissive);
      mat.emissiveIntensity = mesh.userData._origEI;
    } else if (mesh.userData._origColor !== null && mat.color) {
      mat.color.copy(mesh.userData._origColor);
    }
    mat.needsUpdate = true;
  };

  const handleClick = useCallback((e) => {
    e.stopPropagation();
    const target = resolveElementMesh(e.object);
    if (!target) return;
    const match = ELEMENT_NAME_RE.exec(target.name);
    if (!match) return;

    // Clear previous selection highlight
    if (prevRef.current && prevRef.current !== target) {
      clearHighlight(prevRef.current);
    }

    applyHighlight(target);
    prevRef.current = target;

    onElementSelect(match[1], parseInt(match[2], 10));
  }, [onElementSelect]);

  // Clear highlight when EditPanel is closed (selectedMesh set to null)
  useEffect(() => {
    if (!selectedMesh && prevRef.current) {
      clearHighlight(prevRef.current);
      prevRef.current = null;
    }
  }, [selectedMesh]);

  return <primitive object={scene} onClick={handleClick} />;
};

const Viewer = ({ modelUrl, onElementSelect, selectedMesh }) => {
  if (!modelUrl) {
    return (
      <div className="w-full h-full bg-slate-900 flex items-center justify-center">
        <div className="text-center space-y-3 text-slate-500">
          <span className="text-6xl block">🏗️</span>
          <p className="text-xl font-light">No Model Loaded</p>
          <p className="text-sm opacity-60">Upload a floor plan to generate your 3D BIM model</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full bg-slate-900">
      <ViewerErrorBoundary>
        <Canvas shadows dpr={[1, 2]} camera={{ fov: 50, position: [10, 10, 10] }}>
          <Suspense fallback={null}>
            <Stage environment="city" intensity={0.6}>
              <SelectableModel
                url={modelUrl}
                onElementSelect={onElementSelect}
                selectedMesh={selectedMesh}
              />
            </Stage>
          </Suspense>
          <OrbitControls makeDefault />
          <Environment preset="city" />
        </Canvas>
      </ViewerErrorBoundary>
    </div>
  );
};

export default Viewer;
