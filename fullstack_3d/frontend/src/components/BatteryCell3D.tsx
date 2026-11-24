import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export interface LiveCellTelemetry {
  soc: number; // 0..1
  socEst: number; // 0..1
  tempC: number;
  ambientTempC: number;
  cRate: number;
  voltageV: number;
  strategy: "CC-CV" | "Adaptive_ML";
}

interface BatteryCell3DProps {
  chemistry: "li_ion" | "na_ion";
  cccv: LiveCellTelemetry;
  adaptive: LiveCellTelemetry;
  compareMode: boolean;
  xrayCutaway: boolean;
  autoRotate: boolean;
}

interface CellRig {
  group: THREE.Group;
  fillMesh: THREE.Mesh;
  fillMat: THREE.MeshStandardMaterial;
  shellMat: THREE.MeshPhysicalMaterial;
  thermalRingMat: THREE.MeshBasicMaterial;
  platingRing: THREE.Mesh;
  platingRingMat: THREE.MeshBasicMaterial;
  particles: THREE.Points;
  particleBase: Float32Array;
  particleSpeeds: Float32Array;
  maxFillHeight: number;
  fillBaseY: number;
  innerRadiusX: number;
  innerRadiusZ: number;
}

function thermalColor(tempC: number, ambC: number): THREE.Color {
  // Smooth color interpolation: cool teal (dT=0) -> warm amber (dT=4.5) -> hot crimson (dT>=8)
  const dt = Math.max(0, tempC - ambC);
  const t = Math.min(1, dt / 7.5);
  const cCool = new THREE.Color("#14b8a6");
  const cMid = new THREE.Color("#f59e0b");
  const cHot = new THREE.Color("#f43f5e");
  if (t < 0.5) {
    return cCool.clone().lerp(cMid, t * 2.0);
  }
  return cMid.clone().lerp(cHot, (t - 0.5) * 2.0);
}

function buildCellRig(
  chemistry: "li_ion" | "na_ion",
  accentHex: string,
  xrayCutaway: boolean
): CellRig {
  const group = new THREE.Group();
  const isLi = chemistry === "li_ion";
  const bodyColor = isLi ? "#1d4ed8" : "#b45309";
  const ionColor = isLi ? "#38bdf8" : "#fbbf24";

  const shellMat = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(bodyColor),
    metalness: 0.55,
    roughness: 0.25,
    clearcoat: 0.65,
    clearcoatRoughness: 0.18,
    transparent: true,
    opacity: xrayCutaway ? 0.34 : 0.86,
    side: THREE.DoubleSide,
  });

  const metalMat = new THREE.MeshStandardMaterial({
    color: "#cbd5e1",
    metalness: 0.88,
    roughness: 0.22,
  });

  const copperMat = new THREE.MeshStandardMaterial({
    color: "#d97706",
    metalness: 0.85,
    roughness: 0.28,
  });

  const fillMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(accentHex),
    emissive: new THREE.Color(accentHex),
    emissiveIntensity: 0.45,
    roughness: 0.3,
    metalness: 0.2,
    transparent: true,
    opacity: 0.82,
  });

  let fillMesh: THREE.Mesh;
  const maxFillHeight = 2.75;
  const fillBaseY = -1.35;
  let innerRadiusX = 0.72;
  let innerRadiusZ = 0.72;

  if (isLi) {
    // --- Cylindrical 18650 Lithium-Ion Cell ---
    const thetaLen = xrayCutaway ? Math.PI * 1.48 : Math.PI * 2;
    const outerCyl = new THREE.Mesh(
      new THREE.CylinderGeometry(0.92, 0.92, 3.1, 48, 1, true, Math.PI * 0.26, thetaLen),
      shellMat
    );
    group.add(outerCyl);

    // Bottom negative copper collector plate
    const bottomCap = new THREE.Mesh(new THREE.CylinderGeometry(0.92, 0.92, 0.14, 48), copperMat);
    bottomCap.position.y = -1.58;
    group.add(bottomCap);

    // Top crimp collar + positive button cap
    const topRing = new THREE.Mesh(new THREE.CylinderGeometry(0.92, 0.92, 0.12, 48), metalMat);
    topRing.position.y = 1.58;
    group.add(topRing);

    const posCap = new THREE.Mesh(new THREE.CylinderGeometry(0.36, 0.42, 0.22, 32), metalMat);
    posCap.position.y = 1.72;
    group.add(posCap);

    // Internal wound jelly-roll electrode rings (visible in cutaway)
    const jellyMat = new THREE.MeshStandardMaterial({
      color: "#334155",
      metalness: 0.4,
      roughness: 0.6,
      transparent: true,
      opacity: 0.35,
      wireframe: true,
    });
    for (const r of [0.35, 0.56, 0.76]) {
      const coil = new THREE.Mesh(new THREE.CylinderGeometry(r, r, 2.75, 24, 6, true), jellyMat);
      group.add(coil);
    }

    // Dynamic SoC glowing fill cylinder
    fillMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.74, 0.74, 1.0, 36), fillMat);
    fillMesh.position.y = fillBaseY + 0.5;
    group.add(fillMesh);
  } else {
    // --- Prismatic Sodium-Ion Cell (3.2 Ah / 10 Ah HiNa / Transimage Form Factor) ---
    innerRadiusX = 0.88;
    innerRadiusZ = 0.46;

    // Back, left, right, bottom, top anodized prismatic walls (front open in cutaway)
    const backWall = new THREE.Mesh(new THREE.BoxGeometry(2.1, 3.1, 0.08), shellMat);
    backWall.position.z = -0.62;
    group.add(backWall);

    const frontWall = new THREE.Mesh(
      new THREE.BoxGeometry(2.1, 3.1, 0.06),
      new THREE.MeshPhysicalMaterial({
        color: "#fde68a",
        transparent: true,
        opacity: xrayCutaway ? 0.12 : 0.78,
        roughness: 0.1,
        metalness: 0.2,
      })
    );
    frontWall.position.z = 0.62;
    group.add(frontWall);

    for (const xSign of [-1, 1]) {
      const side = new THREE.Mesh(new THREE.BoxGeometry(0.08, 3.1, 1.3), shellMat);
      side.position.x = xSign * 1.05;
      group.add(side);
    }

    const basePlate = new THREE.Mesh(new THREE.BoxGeometry(2.16, 0.14, 1.34), metalMat);
    basePlate.position.y = -1.58;
    group.add(basePlate);

    const topPlate = new THREE.Mesh(new THREE.BoxGeometry(2.16, 0.14, 1.34), metalMat);
    topPlate.position.y = 1.58;
    group.add(topPlate);

    // Dual busbar terminals (+ copper, - steel) + central safety vent
    const posPost = new THREE.Mesh(new THREE.BoxGeometry(0.38, 0.26, 0.38), copperMat);
    posPost.position.set(-0.62, 1.76, 0);
    group.add(posPost);

    const negPost = new THREE.Mesh(new THREE.BoxGeometry(0.38, 0.26, 0.38), metalMat);
    negPost.position.set(0.62, 1.76, 0);
    group.add(negPost);

    const vent = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.18, 0.06, 20), metalMat);
    vent.position.set(0, 1.67, 0);
    group.add(vent);

    // Stacked electrode plates inside cutaway
    const plateMat = new THREE.MeshStandardMaterial({
      color: "#475569",
      metalness: 0.5,
      roughness: 0.5,
      transparent: true,
      opacity: 0.25,
    });
    for (let z = -0.4; z <= 0.41; z += 0.2) {
      const plate = new THREE.Mesh(new THREE.BoxGeometry(1.86, 2.75, 0.025), plateMat);
      plate.position.z = z;
      group.add(plate);
    }

    // Dynamic SoC glowing fill box
    fillMesh = new THREE.Mesh(new THREE.BoxGeometry(1.82, 1.0, 1.02), fillMat);
    fillMesh.position.y = fillBaseY + 0.5;
    group.add(fillMesh);
  }

  // Ground Thermal Aura Ring
  const thermalRingMat = new THREE.MeshBasicMaterial({
    color: "#14b8a6",
    transparent: true,
    opacity: 0.55,
    side: THREE.DoubleSide,
  });
  const thermalRing = new THREE.Mesh(new THREE.RingGeometry(1.08, 1.32, 48), thermalRingMat);
  thermalRing.rotation.x = -Math.PI / 2;
  thermalRing.position.y = -1.67;
  group.add(thermalRing);

  // Upper Anode Plating / Over-Stress Risk Halo
  const platingRingMat = new THREE.MeshBasicMaterial({
    color: "#f43f5e",
    transparent: true,
    opacity: 0.0,
    side: THREE.DoubleSide,
  });
  const platingRing = new THREE.Mesh(new THREE.TorusGeometry(isLi ? 0.99 : 1.18, 0.035, 12, 48), platingRingMat);
  platingRing.rotation.x = Math.PI / 2;
  platingRing.position.y = 1.15;
  group.add(platingRing);

  // Real-time Ion Flux Particle System (Li+ or Na+ migrating during charge)
  const count = 160;
  const positions = new Float32Array(count * 3);
  const baseCoords = new Float32Array(count * 3);
  const speeds = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const angle = Math.random() * Math.PI * 2;
    const rad = Math.sqrt(Math.random());
    const x = isLi ? Math.cos(angle) * rad * innerRadiusX : (Math.random() * 2 - 1) * innerRadiusX;
    const z = isLi ? Math.sin(angle) * rad * innerRadiusZ : (Math.random() * 2 - 1) * innerRadiusZ;
    const y = fillBaseY + Math.random() * maxFillHeight;
    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;
    baseCoords[i * 3] = x;
    baseCoords[i * 3 + 1] = y;
    baseCoords[i * 3 + 2] = z;
    speeds[i] = 0.65 + Math.random() * 0.75;
  }

  const pGeo = new THREE.BufferGeometry();
  pGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const pMat = new THREE.PointsMaterial({
    color: new THREE.Color(ionColor),
    size: 0.085,
    transparent: true,
    opacity: 0.92,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const particles = new THREE.Points(pGeo, pMat);
  group.add(particles);

  return {
    group,
    fillMesh,
    fillMat,
    shellMat,
    thermalRingMat,
    platingRing,
    platingRingMat,
    particles,
    particleBase: baseCoords,
    particleSpeeds: speeds,
    maxFillHeight,
    fillBaseY,
    innerRadiusX,
    innerRadiusZ,
  };
}

function updateCellRig(rig: CellRig, tel: LiveCellTelemetry, dt: number, elapsed: number) {
  const socClamped = Math.max(0.05, Math.min(1.0, tel.soc));
  const h = socClamped * rig.maxFillHeight;
  rig.fillMesh.scale.y = h;
  rig.fillMesh.position.y = rig.fillBaseY + h * 0.5;

  // Dynamic thermal glow color driven by live temperature rise
  const tColor = thermalColor(tel.tempC, tel.ambientTempC);
  rig.fillMat.color.copy(tColor);
  rig.fillMat.emissive.copy(tColor);
  rig.fillMat.emissiveIntensity = 0.35 + Math.min(0.65, Math.max(0, tel.tempC - tel.ambientTempC) * 0.09);
  rig.thermalRingMat.color.copy(tColor);

  // High-SoC + High-C-rate plating stress warning halo
  const platingStress = Math.max(0, tel.soc - 0.64) * Math.max(0, tel.cRate - 0.52) * 3.8;
  const clampedStress = Math.min(1.0, platingStress);
  rig.platingRingMat.opacity = clampedStress > 0.06 ? clampedStress * (0.55 + 0.35 * Math.sin(elapsed * 7.0)) : 0.0;
  rig.platingRing.scale.setScalar(1.0 + 0.04 * Math.sin(elapsed * 5.0));

  // Animate Li+/Na+ ion stream velocity proportional to live charging C-rate
  const posAttr = rig.particles.geometry.getAttribute("position") as THREE.BufferAttribute;
  const arr = posAttr.array as Float32Array;
  const flowSpeed = Math.max(0.08, tel.cRate) * 1.45;
  const topY = rig.fillBaseY + rig.maxFillHeight;

  for (let i = 0; i < rig.particleSpeeds.length; i++) {
    let y = arr[i * 3 + 1] + dt * flowSpeed * rig.particleSpeeds[i];
    if (y > topY) {
      y = rig.fillBaseY;
    }
    arr[i * 3 + 1] = y;
    // Gentle helical shimmer
    const phase = elapsed * 1.5 + i * 0.37;
    arr[i * 3] = rig.particleBase[i * 3] + Math.sin(phase) * 0.025;
  }
  posAttr.needsUpdate = true;
}

export const BatteryCell3D: React.FC<BatteryCell3DProps> = ({
  chemistry,
  cccv,
  adaptive,
  compareMode,
  xrayCutaway,
  autoRotate,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const telemetryRef = useRef({ cccv, adaptive, autoRotate });
  telemetryRef.current = { cccv, adaptive, autoRotate };

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2("#070c17", 0.045);

    const width = container.clientWidth || 720;
    const height = container.clientHeight || 470;
    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
    camera.position.set(0, 1.65, compareMode ? 7.6 : 5.4);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    container.innerHTML = "";
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.minDistance = 3.2;
    controls.maxDistance = 13.0;
    controls.maxPolarAngle = Math.PI * 0.62;
    controls.target.set(0, 0.05, 0);

    // Studio Precision Lighting
    scene.add(new THREE.HemisphereLight("#bae6fd", "#0f172a", 0.9));
    const keyLight = new THREE.DirectionalLight("#ffffff", 1.7);
    keyLight.position.set(5, 8, 6);
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(chemistry === "li_ion" ? "#38bdf8" : "#f59e0b", 1.1);
    rimLight.position.set(-6, 4, -5);
    scene.add(rimLight);

    // Laboratory Ground Grid & Pedestal Plane
    const grid = new THREE.GridHelper(16, 24, "#1e3a5f", "#0f1c33");
    grid.position.y = -1.72;
    scene.add(grid);

    const floorMesh = new THREE.Mesh(
      new THREE.CircleGeometry(7.5, 64),
      new THREE.MeshStandardMaterial({
        color: "#091121",
        metalness: 0.4,
        roughness: 0.75,
        transparent: true,
        opacity: 0.85,
      })
    );
    floorMesh.rotation.x = -Math.PI / 2;
    floorMesh.position.y = -1.73;
    scene.add(floorMesh);

    // Build Cell Rig(s)
    const adaptiveRig = buildCellRig(chemistry, "#10b981", xrayCutaway);
    let cccvRig: CellRig | null = null;

    if (compareMode) {
      cccvRig = buildCellRig(chemistry, "#f43f5e", xrayCutaway);
      cccvRig.group.position.x = -2.05;
      adaptiveRig.group.position.x = 2.05;
      scene.add(cccvRig.group);
      scene.add(adaptiveRig.group);
    } else {
      adaptiveRig.group.position.x = 0;
      scene.add(adaptiveRig.group);
    }

    const clock = new THREE.Clock();
    let rafId = 0;

    const animate = () => {
      rafId = requestAnimationFrame(animate);
      const dt = Math.min(0.05, clock.getDelta());
      const elapsed = clock.elapsedTime;
      const { cccv: cTel, adaptive: aTel, autoRotate: rotate } = telemetryRef.current;

      controls.autoRotate = rotate;
      controls.autoRotateSpeed = 0.85;
      controls.update();

      updateCellRig(adaptiveRig, aTel, dt, elapsed);
      if (cccvRig) {
        updateCellRig(cccvRig, cTel, dt, elapsed);
      }

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
  }, [chemistry, compareMode, xrayCutaway]);

  return <div ref={containerRef} className="stage-canvas" />;
};
