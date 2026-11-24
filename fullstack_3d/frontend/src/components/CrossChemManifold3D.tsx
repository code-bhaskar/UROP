import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { PcaTaskCloud } from "../types";

interface CrossChemManifold3DProps {
  cloud: PcaTaskCloud;
  morphFactor: number; // 0 = Raw Features, 1 = Physics+OCV+CORAL Aligned
}

export const CrossChemManifold3D: React.FC<CrossChemManifold3DProps> = ({ cloud, morphFactor }) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const morphRef = useRef(morphFactor);
  morphRef.current = morphFactor;

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    const width = container.clientWidth || 640;
    const height = container.clientHeight || 400;
    const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);
    camera.position.set(2.9, 2.2, 3.6);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    container.innerHTML = "";
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.7;

    // Coordinate wireframe cube & subtle axes
    const boxGeo = new THREE.BoxGeometry(2.8, 2.8, 2.8);
    const edges = new THREE.EdgesGeometry(boxGeo);
    const wireCube = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({ color: "#334155", transparent: true, opacity: 0.45 })
    );
    scene.add(wireCube);

    const axes = new THREE.AxesHelper(1.55);
    scene.add(axes);

    const makeCloud = (
      rawPts: [number, number, number][],
      alnPts: [number, number, number][],
      colorHex: string
    ) => {
      const n = Math.min(rawPts.length, alnPts.length);
      const pos = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        pos[i * 3] = rawPts[i][0];
        pos[i * 3 + 1] = rawPts[i][1];
        pos[i * 3 + 2] = rawPts[i][2];
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      const mat = new THREE.PointsMaterial({
        color: new THREE.Color(colorHex),
        size: 0.08,
        transparent: true,
        opacity: 0.88,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      return { pts: new THREE.Points(geo, mat), rawPts, alnPts, n };
    };

    const liCloud = makeCloud(cloud.raw_li, cloud.aligned_li, "#38bdf8");
    const naCloud = makeCloud(cloud.raw_na, cloud.aligned_na, "#f59e0b");
    scene.add(liCloud.pts);
    scene.add(naCloud.pts);

    // Centroid spheres + connecting line visualizing domain shift collapse
    const liCentroid = new THREE.Mesh(
      new THREE.SphereGeometry(0.085, 20, 20),
      new THREE.MeshBasicMaterial({ color: "#7dd3fc" })
    );
    const naCentroid = new THREE.Mesh(
      new THREE.SphereGeometry(0.085, 20, 20),
      new THREE.MeshBasicMaterial({ color: "#fcd34d" })
    );
    scene.add(liCentroid);
    scene.add(naCentroid);

    const lineGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
    const bridgeLine = new THREE.Line(
      lineGeo,
      new THREE.LineBasicMaterial({ color: "#10b981", transparent: true, opacity: 0.9 })
    );
    scene.add(bridgeLine);

    let smoothedT = morphRef.current;
    let rafId = 0;

    const updateCloud = (c: ReturnType<typeof makeCloud>, t: number): THREE.Vector3 => {
      const attr = c.pts.geometry.getAttribute("position") as THREE.BufferAttribute;
      const arr = attr.array as Float32Array;
      let cx = 0,
        cy = 0,
        cz = 0;
      for (let i = 0; i < c.n; i++) {
        const x = c.rawPts[i][0] * (1 - t) + c.alnPts[i][0] * t;
        const y = c.rawPts[i][1] * (1 - t) + c.alnPts[i][1] * t;
        const z = c.rawPts[i][2] * (1 - t) + c.alnPts[i][2] * t;
        arr[i * 3] = x;
        arr[i * 3 + 1] = y;
        arr[i * 3 + 2] = z;
        cx += x;
        cy += y;
        cz += z;
      }
      attr.needsUpdate = true;
      return new THREE.Vector3(cx / Math.max(1, c.n), cy / Math.max(1, c.n), cz / Math.max(1, c.n));
    };

    const animate = () => {
      rafId = requestAnimationFrame(animate);
      smoothedT += (morphRef.current - smoothedT) * 0.12;

      const cLi = updateCloud(liCloud, smoothedT);
      const cNa = updateCloud(naCloud, smoothedT);
      liCentroid.position.copy(cLi);
      naCentroid.position.copy(cNa);

      const linePos = bridgeLine.geometry.getAttribute("position") as THREE.BufferAttribute;
      linePos.setXYZ(0, cLi.x, cLi.y, cLi.z);
      linePos.setXYZ(1, cNa.x, cNa.y, cNa.z);
      linePos.needsUpdate = true;

      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const ro = new ResizeObserver(() => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      if (w > 0 && h > 0) {
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
      }
    });
    ro.observe(container);

    return () => {
      cancelAnimationFrame(rafId);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
    };
  }, [cloud]);

  return <div ref={containerRef} style={{ width: "100%", height: "390px", cursor: "grab" }} />;
};
