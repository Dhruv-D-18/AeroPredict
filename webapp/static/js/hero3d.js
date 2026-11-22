/* =====================================================================
   AeroPredict — 3D hero
   A procedurally built twin-engine airliner hovering over a pale runway.
   No external model files: geometry + canvas textures only (~0 KB assets).
   ===================================================================== */
import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const hero   = document.getElementById('hero');
const mount  = document.getElementById('heroCanvas');
const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function webglAvailable() {
    try {
        const c = document.createElement('canvas');
        return !!(window.WebGLRenderingContext && (c.getContext('webgl2') || c.getContext('webgl')));
    } catch (e) { return false; }
}

if (!mount || !webglAvailable()) {
    hero && hero.classList.add('no-webgl');
} else {
    try { init(); } catch (err) { console.warn('3D hero disabled:', err); hero.classList.add('no-webgl'); }
}

function init() {
    /* ---------- Renderer / scene / camera ---------- */
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.setClearColor(0x000000, 0);
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    pmrem.dispose();

    const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 400);
    const HEADING = -1.95;                               // nose pointing toward the viewer, slightly left
    const camBase = new THREE.Vector3(13.6, 10.4, 23.6);
    const lookAt  = new THREE.Vector3(-0.6, 1.5, 0);

    /* ---------- Lights ---------- */
    scene.add(new THREE.HemisphereLight(0xF4F9FF, 0xDCE6EF, 0.9));
    const sun = new THREE.DirectionalLight(0xFFFFFF, 2.1);
    sun.position.set(-6, 14, 8);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    sun.shadow.camera.left = -9; sun.shadow.camera.right = 9;
    sun.shadow.camera.top = 9;   sun.shadow.camera.bottom = -9;
    sun.shadow.camera.near = 1;  sun.shadow.camera.far = 40;
    sun.shadow.radius = 6;
    sun.shadow.bias = -0.0004;
    scene.add(sun);
    const rim = new THREE.DirectionalLight(0xCFE4F7, 0.8);
    rim.position.set(8, 4, -10);
    scene.add(rim);

    /* ---------- Ground & runway ---------- */
    const runwayTex = makeRunwayTexture(renderer);
    const groundGeo = new THREE.PlaneGeometry(100, 100);
    groundGeo.rotateX(-Math.PI / 2);
    const ground = new THREE.Mesh(groundGeo, new THREE.MeshStandardMaterial({
        map: runwayTex.map, alphaMap: runwayTex.alpha, transparent: true,
        roughness: 0.95, metalness: 0, depthWrite: false,
    }));
    ground.rotation.y = HEADING + Math.PI / 2;   // runway runs along the aircraft's heading
    scene.add(ground);

    const shadowGeo = new THREE.PlaneGeometry(40, 40);
    shadowGeo.rotateX(-Math.PI / 2);
    const shadowCatcher = new THREE.Mesh(shadowGeo, new THREE.ShadowMaterial({ opacity: 0.075 }));
    shadowCatcher.position.y = 0.01;
    shadowCatcher.receiveShadow = true;
    scene.add(shadowCatcher);

    /* ---------- Aircraft ---------- */
    const aircraft = buildAircraft(renderer);
    const rig = new THREE.Group();          // handles heading
    rig.add(aircraft);
    rig.rotation.y = HEADING;
    scene.add(rig);

    /* ---------- Layout & resize ---------- */
    let width = 1, height = 1, wide = true;
    function resize() {
        const r = mount.getBoundingClientRect();
        width = Math.max(1, r.width); height = Math.max(1, r.height);
        wide = window.innerWidth > 860;       // matches the CSS layout breakpoint
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.fov = wide ? 27 : 24;
        // Wide: shift the frame so the aircraft sits to the right of the copy.
        // Narrow (mobile, canvas below the copy): tighter framing, centred.
        if (wide) camera.setViewOffset(width, height, -width * 0.17, -height * 0.03, width, height);
        else      camera.setViewOffset(width, height, width * 0.06, height * 0.02, width, height);
        camera.updateProjectionMatrix();
        render();
    }

    /* ---------- Interaction ---------- */
    const pointer = { x: 0, y: 0 }, smooth = { x: 0, y: 0 };
    window.addEventListener('pointermove', (e) => {
        pointer.x = (e.clientX / window.innerWidth) * 2 - 1;
        pointer.y = (e.clientY / window.innerHeight) * 2 - 1;
    }, { passive: true });

    /* ---------- Loop ---------- */
    const clock = new THREE.Clock();
    let running = false, visible = true, raf = 0, t = 0;

    function update(dt) {
        t += dt;
        smooth.x += (pointer.x - smooth.x) * Math.min(1, dt * 2.2);
        smooth.y += (pointer.y - smooth.y) * Math.min(1, dt * 2.2);

        // Gentle float + banking
        aircraft.position.y = 2.15 + Math.sin(t * 0.7) * 0.12;
        aircraft.rotation.x = Math.sin(t * 0.45) * 0.03 + smooth.x * 0.04;      // roll
        aircraft.rotation.z = 0.035 + Math.sin(t * 0.55 + 1.1) * 0.012;         // pitch (nose slightly up)
        rig.rotation.y = HEADING + Math.sin(t * 0.18) * 0.05 + smooth.x * 0.08;

        // Slow cinematic camera drift + parallax
        const drift = Math.sin(t * 0.12) * 0.9;
        camera.position.set(
            camBase.x + drift + smooth.x * 1.4,
            camBase.y - smooth.y * 0.7 + Math.sin(t * 0.21) * 0.15,
            camBase.z + Math.cos(t * 0.12) * 0.5
        );
        camera.lookAt(lookAt);

        // Runway slides underneath → sense of forward motion
        runwayTex.map.offset.y = (runwayTex.map.offset.y + dt * 0.018) % 1;
    }

    function render() { renderer.render(scene, camera); }

    function frame() {
        if (!running) return;
        raf = requestAnimationFrame(frame);
        update(Math.min(clock.getDelta(), 0.05));
        render();
    }
    function start() { if (running || reduce) return; running = true; clock.getDelta(); frame(); }
    function stop()  { running = false; cancelAnimationFrame(raf); }

    new ResizeObserver(resize).observe(mount);
    resize();
    update(0); render();

    if ('IntersectionObserver' in window) {
        new IntersectionObserver(([e]) => { visible = e.isIntersecting; visible && !document.hidden ? start() : stop(); },
            { threshold: 0.02 }).observe(mount);
    }
    document.addEventListener('visibilitychange', () => (document.hidden || !visible) ? stop() : start());
    start();

    requestAnimationFrame(() => hero.classList.add('webgl-ready'));
}

/* =====================================================================
   Aircraft construction
   Axes: nose → +X, up → +Y, right wing → +Z
   ===================================================================== */
function buildAircraft(renderer) {
    const plane = new THREE.Group();
    const L = 12.4, R = 0.66;
    const aniso = renderer.capabilities.getMaxAnisotropy();

    // ---- Materials ----
    const livery = makeLiveryTexture(aniso);
    const fuselageMat = new THREE.MeshPhysicalMaterial({
        map: livery, roughness: 0.32, metalness: 0.08, clearcoat: 0.7, clearcoatRoughness: 0.18,
    });
    const wingMat   = new THREE.MeshPhysicalMaterial({ color: 0xDCE3EA, roughness: 0.42, metalness: 0.35, clearcoat: 0.3 });
    const nacelleMat= new THREE.MeshPhysicalMaterial({ color: 0xF2F5F8, roughness: 0.3, metalness: 0.1, clearcoat: 0.8, clearcoatRoughness: 0.15, side: THREE.DoubleSide });
    const lipMat    = new THREE.MeshStandardMaterial({ color: 0xC3CCD5, roughness: 0.25, metalness: 0.9 });
    const intakeMat = new THREE.MeshStandardMaterial({ color: 0x1A2632, roughness: 0.55, metalness: 0.4 });
    const fanMat    = new THREE.MeshStandardMaterial({ color: 0x5E6B78, roughness: 0.35, metalness: 0.85 });
    const navyMat   = new THREE.MeshPhysicalMaterial({ color: 0x12304A, roughness: 0.35, metalness: 0.1, clearcoat: 0.6 });

    // ---- Fuselage (custom lofted tube) ----
    const fuselage = new THREE.Mesh(buildFuselage(L, R), fuselageMat);
    fuselage.castShadow = true;
    plane.add(fuselage);

    // ---- Wings ----
    const wingShape = planform([
        [1.55, 0.0], [ -1.95, 5.75], [-2.55, 5.75], [-1.75, 1.6], [-1.65, 0.0],
    ]);
    const wingGeo = extrudeFlat(wingShape, 0.085, 0.045);
    [1, -1].forEach((side) => {
        const w = new THREE.Mesh(wingGeo, wingMat);
        w.position.set(0.4, -R * 0.42, 0);
        w.scale.z = side;
        w.rotation.x = side * -0.085;     // dihedral
        w.castShadow = true;
        plane.add(w);

        // Winglet
        const wl = new THREE.Mesh(extrudeFlat(planform([[0, 0], [-0.55, 0.95], [-0.8, 0.95], [-0.6, 0]]), 0.04, 0.02), navyMat);
        wl.rotation.x = -Math.PI / 2;     // planform z → world up
        wl.position.set(0.4 - 1.95 - 0.02, -R * 0.42 + 5.75 * Math.sin(0.085) + 0.02, side * 5.75);
        wl.castShadow = true;
        plane.add(wl);

        // Engine
        const eng = buildEngine(nacelleMat, lipMat, intakeMat, fanMat, wingMat);
        eng.position.set(0.95, -R * 0.42 - 0.62 + 1.95 * Math.sin(0.085) * 0.8, side * 1.95);
        plane.add(eng);
    });

    // ---- Wing-body fairing ----
    const fairing = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 16), new THREE.MeshPhysicalMaterial({ color: 0xE6ECF1, roughness: 0.35, metalness: 0.1, clearcoat: 0.5 }));
    fairing.scale.set(2.6, 0.42, 0.72);
    fairing.position.set(-0.3, -R * 0.6, 0);
    fairing.castShadow = true;
    plane.add(fairing);

    // ---- Horizontal stabiliser ----
    const hGeo = extrudeFlat(planform([[0.0, 0.0], [-1.25, 2.15], [-1.62, 2.15], [-1.25, 0.0]]), 0.05, 0.025);
    [1, -1].forEach((side) => {
        const h = new THREE.Mesh(hGeo, wingMat);
        h.position.set(-L / 2 + 1.75, R * 0.22, 0);
        h.scale.z = side;
        h.rotation.x = side * -0.12;
        h.castShadow = true;
        plane.add(h);
    });

    // ---- Vertical fin ----
    const finShape = new THREE.Shape();
    const fin = [[0.0, 0.0], [-1.85, 2.55], [-2.55, 2.55], [-2.25, 0.0]];
    finShape.moveTo(fin[0][0], fin[0][1]);
    fin.slice(1).forEach(([x, y]) => finShape.lineTo(x, y));
    finShape.closePath();
    const finGeo = new THREE.ExtrudeGeometry(finShape, { depth: 0.09, bevelEnabled: true, bevelThickness: 0.03, bevelSize: 0.03, bevelSegments: 2, curveSegments: 1 });
    finGeo.translate(0, 0, -0.045);
    const finTex = makeFinTexture(aniso);
    finTex.repeat.set(1 / 2.6, 1 / 2.6);
    finTex.offset.set(2.6 / 2.6, 0);   // shape x spans -2.55..0  → u 0..1
    const finMesh = new THREE.Mesh(finGeo, [
        new THREE.MeshPhysicalMaterial({ map: finTex, roughness: 0.32, metalness: 0.1, clearcoat: 0.7 }),
        navyMat,
    ]);
    finMesh.position.set(-L / 2 + 2.35, R * 0.62, 0);
    finMesh.castShadow = true;
    plane.add(finMesh);

    plane.traverse((o) => { if (o.isMesh) o.receiveShadow = false; });
    return plane;
}

/* Lofted fuselage: rings along X with radius + centreline profiles. */
function buildFuselage(L, R) {
    const N = 140, M = 56;
    const pos = [], uv = [], idx = [];
    for (let i = 0; i <= N; i++) {
        const t = i / N;                         // 0 = tail, 1 = nose
        const x = -L / 2 + t * L;
        let r = R, cy = 0;
        if (t > 0.84) {                          // nose
            const s = (t - 0.84) / 0.16;
            r = R * Math.pow(Math.max(0, 1 - Math.pow(s, 2.3)), 0.5);
            cy = -R * 0.16 * s * s;
        } else if (t < 0.24) {                   // tail cone with upsweep
            const s = (0.24 - t) / 0.24;
            r = R * (1 - 0.86 * Math.pow(s, 1.35));
            cy = R * 0.62 * Math.pow(s, 1.25);
        }
        r = Math.max(r, 0.004);
        for (let j = 0; j <= M; j++) {
            const v = j / M;
            const phi = Math.PI + v * Math.PI * 2;  // v=0 belly, 0.5 crown
            const y = cy + Math.cos(phi) * r * 1.05;
            const z = Math.sin(phi) * r;
            pos.push(x, y, z);
            uv.push(t, v);
        }
    }
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < M; j++) {
            const a = i * (M + 1) + j, b = a + M + 1;
            idx.push(a, a + 1, b, b, a + 1, b + 1);
        }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    g.setIndex(idx);
    g.computeVertexNormals();
    return g;
}

function buildEngine(nacelleMat, lipMat, intakeMat, fanMat, pylonMat) {
    const g = new THREE.Group();
    const len = 1.7, r = 0.4;
    // Nacelle profile (radius vs x), lathed around X
    const prof = [
        [0.00, r * 0.86], [0.04, r * 0.97], [0.12, r * 1.0], [0.45, r * 1.0],
        [0.75, r * 0.9], [1.00, r * 0.66],
    ].map(([u, rr]) => new THREE.Vector2(rr, (0.5 - u) * len));
    const lathe = new THREE.LatheGeometry(prof, 48);
    lathe.rotateZ(-Math.PI / 2);
    const nacelle = new THREE.Mesh(lathe, nacelleMat);
    nacelle.castShadow = true;
    g.add(nacelle);

    const lip = new THREE.Mesh(new THREE.TorusGeometry(r * 0.9, 0.035, 12, 48), lipMat);
    lip.rotation.y = Math.PI / 2;
    lip.position.x = len * 0.5 - 0.01;
    g.add(lip);

    const intake = new THREE.Mesh(new THREE.CircleGeometry(r * 0.88, 48), intakeMat);
    intake.rotation.y = Math.PI / 2;
    intake.position.x = len * 0.5 - 0.09;
    g.add(intake);

    const fan = new THREE.Mesh(new THREE.CircleGeometry(r * 0.8, 24), fanMat);
    fan.rotation.y = Math.PI / 2;
    fan.position.x = len * 0.5 - 0.1;
    g.add(fan);

    const spinner = new THREE.Mesh(new THREE.ConeGeometry(0.13, 0.3, 24), lipMat);
    spinner.rotation.z = -Math.PI / 2;
    spinner.position.x = len * 0.5 - 0.02;
    g.add(spinner);

    const exhaust = new THREE.Mesh(new THREE.ConeGeometry(r * 0.5, 0.5, 32), lipMat);
    exhaust.rotation.z = Math.PI / 2;
    exhaust.position.x = -len * 0.5 - 0.18;
    g.add(exhaust);

    const pylon = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.42, 0.1), pylonMat);
    pylon.position.set(-0.25, 0.48, 0);
    pylon.castShadow = true;
    g.add(pylon);
    return g;
}

/* Planform helpers: shape drawn in (x, z), extruded thin along Y. */
function planform(pts) {
    const s = new THREE.Shape();
    s.moveTo(pts[0][0], pts[0][1]);
    pts.slice(1).forEach(([x, z]) => s.lineTo(x, z));
    s.closePath();
    return s;
}
function extrudeFlat(shape, thickness, bevel) {
    const g = new THREE.ExtrudeGeometry(shape, {
        depth: thickness, bevelEnabled: true, bevelThickness: bevel, bevelSize: bevel, bevelSegments: 2, curveSegments: 1,
    });
    g.rotateX(Math.PI / 2);          // shape Y → world Z, depth → world −Y
    g.translate(0, thickness / 2, 0);
    return g;
}

/* =====================================================================
   Canvas textures
   ===================================================================== */
function makeLiveryTexture(aniso) {
    const W = 2048, H = 1024;
    const c = document.createElement('canvas'); c.width = W; c.height = H;
    const g = c.getContext('2d');
    // canvas y = (1 - v) * H  (flipY). Livery is symmetric so both sides match.
    const vy = (v) => (1 - v) * H;

    // Base: white crown → soft grey belly
    const grad = g.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0.00, '#D6DEE6');
    grad.addColorStop(0.16, '#EEF2F6');
    grad.addColorStop(0.30, '#FBFCFD');
    grad.addColorStop(0.50, '#FFFFFF');
    grad.addColorStop(0.70, '#FBFCFD');
    grad.addColorStop(0.84, '#EEF2F6');
    grad.addColorStop(1.00, '#D6DEE6');
    g.fillStyle = grad; g.fillRect(0, 0, W, H);

    const sides = [0.25, 0.75];         // v of each flank
    sides.forEach((vc) => {
        const dir = vc < 0.5 ? 1 : -1;   // towards the crown
        // Navy cheatline + aviation-blue pinstripe below the windows
        const yN = vy(vc - dir * 0.035);
        g.fillStyle = '#12304A';
        g.fillRect(W * 0.17, yN - 9, W * 0.71, 18);
        g.fillStyle = '#1769AA';
        g.fillRect(W * 0.17, yN + dir * 20 - 3, W * 0.66, 6);

        // Passenger windows
        const yw = vy(vc + dir * 0.06);
        g.fillStyle = '#1E3347';
        for (let x = W * 0.25; x < W * 0.83; x += 21.5) {
            if (Math.abs(x - W * 0.5) < 14 || Math.abs(x - W * 0.62) < 12) continue; // over-wing exits
            roundRect(g, x, yw - 8, 10, 16, 5); g.fill();
        }
        // Doors
        g.strokeStyle = 'rgba(18,48,74,0.28)'; g.lineWidth = 2.2;
        [0.215, 0.865].forEach((u) => { roundRect(g, W * u, yw - 34, 26, 60, 8); g.stroke(); });

        // Cockpit windscreen
        g.fillStyle = '#1A2B3C';
        const yc = vy(vc + dir * 0.115);
        for (let k = 0; k < 3; k++) {
            g.beginPath();
            const x0 = W * 0.936 + k * 13;
            g.moveTo(x0, yc - 13); g.lineTo(x0 + 11, yc - 13 + (k * 3)); g.lineTo(x0 + 11, yc + 6); g.lineTo(x0, yc + 8);
            g.closePath(); g.fill();
        }
    });

    // Belly tint near the nose radome
    const nose = g.createLinearGradient(W * 0.97, 0, W, 0);
    nose.addColorStop(0, 'rgba(214,222,230,0)');
    nose.addColorStop(1, 'rgba(200,210,220,0.9)');
    g.fillStyle = nose; g.fillRect(W * 0.97, 0, W * 0.03, H);

    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = aniso;
    return tex;
}

function makeFinTexture(aniso) {
    const S = 512;
    const c = document.createElement('canvas'); c.width = S; c.height = S;
    const g = c.getContext('2d');
    g.fillStyle = '#12304A'; g.fillRect(0, 0, S, S);
    // Sweeping aviation-blue band
    g.fillStyle = '#1769AA';
    g.beginPath(); g.moveTo(0, S * 0.78); g.quadraticCurveTo(S * 0.5, S * 0.62, S, S * 0.18);
    g.lineTo(S, S * 0.36); g.quadraticCurveTo(S * 0.55, S * 0.78, 0, S * 0.96); g.closePath(); g.fill();
    g.fillStyle = '#8CC4EC';
    g.beginPath(); g.moveTo(0, S * 0.98); g.quadraticCurveTo(S * 0.55, S * 0.82, S, S * 0.4);
    g.lineTo(S, S * 0.44); g.quadraticCurveTo(S * 0.56, S * 0.86, 0, S * 1.02); g.closePath(); g.fill();
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = aniso;
    tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
    return tex;
}

function makeRunwayTexture(renderer) {
    // Colour map tiles along the runway (V); alpha map fades everything into the page background.
    const W = 1024, H = 2048;
    const c = document.createElement('canvas'); c.width = W; c.height = H;
    const g = c.getContext('2d');
    g.fillStyle = '#EEF3F8'; g.fillRect(0, 0, W, H);
    // subtle grass/apron texture lines
    g.strokeStyle = 'rgba(23,105,170,0.05)'; g.lineWidth = 2;
    for (let x = 0; x < W; x += 64) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke(); }
    // Taxiway (parallel, thin)
    g.fillStyle = '#E4EBF2'; g.fillRect(W * 0.70, 0, W * 0.06, H);
    g.fillStyle = 'rgba(201,138,27,0.45)'; g.fillRect(W * 0.728, 0, 3, H);
    // Runway
    const rx = W * 0.40, rw = W * 0.2;
    g.fillStyle = "#D6DFE8"; g.fillRect(rx, 0, rw, H);
    g.fillStyle = '#FFFFFF';
    g.fillRect(rx + 6, 0, 4, H); g.fillRect(rx + rw - 10, 0, 4, H);          // edge lines
    for (let y = 0; y < H; y += 128) g.fillRect(W * 0.5 - 3, y, 6, 70);       // centreline
    // Touchdown zone bars
    for (let k = 0; k < 3; k++) {
        const y = 300 + k * 90;
        g.fillRect(rx + 30, y, 16, 54); g.fillRect(rx + 54, y, 16, 54);
        g.fillRect(rx + rw - 46, y, 16, 54); g.fillRect(rx + rw - 70, y, 16, 54);
    }
    // Edge lights
    g.fillStyle = 'rgba(23,105,170,0.55)';
    for (let y = 30; y < H; y += 96) { g.fillRect(rx - 10, y, 5, 5); g.fillRect(rx + rw + 5, y, 5, 5); }

    const map = new THREE.CanvasTexture(c);
    map.colorSpace = THREE.SRGBColorSpace;
    map.wrapS = THREE.ClampToEdgeWrapping; map.wrapT = THREE.RepeatWrapping;
    map.repeat.set(1, 3);
    map.anisotropy = renderer.capabilities.getMaxAnisotropy();

    // Radial alpha: opaque centre → transparent edge
    const a = document.createElement('canvas'); a.width = a.height = 512;
    const ag = a.getContext('2d');
    const rad = ag.createRadialGradient(256, 256, 20, 256, 256, 256);
    rad.addColorStop(0, '#fff'); rad.addColorStop(0.28, '#bbb'); rad.addColorStop(0.62, '#222'); rad.addColorStop(1, '#000');
    ag.fillStyle = rad; ag.fillRect(0, 0, 512, 512);
    const alpha = new THREE.CanvasTexture(a);
    return { map, alpha };
}

function roundRect(g, x, y, w, h, r) {
    g.beginPath();
    g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
    g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
}
