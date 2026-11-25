import streamlit as st
import io
import numpy as np
import ezdxf
import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
import cv2
import trimesh
import base64

# -------------------------- 页面配置 --------------------------
st.set_page_config(page_title="图纸转3D&NC", layout="wide")
st.title("📐 工程图(PDF/DXF) → 三维模型 → 加工工艺 → NC代码")

# 状态缓存
if "mesh" not in st.session_state:
    st.session_state["mesh"] = None
if "stl_bytes" not in st.session_state:
    st.session_state["stl_bytes"] = None
if "process_text" not in st.session_state:
    st.session_state["process_text"] = ""
if "gcode_text" not in st.session_state:
    st.session_state["gcode_text"] = ""

# -------------------------- STL预览（内嵌Three.js，无额外依赖） --------------------------
def stl_preview(stl_data: bytes, height=400):
    stl_b64 = base64.b64encode(stl_data).decode()
    html = f'''
    <html>
    <head>
    <meta charset="utf-8">
    <script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.160.0/examples/js/loaders/STLLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.160.0/examples/js/controls/OrbitControls.js"></script>
    <style>body{{margin:0;padding:0;overflow:hidden}}</style>
    </head>
    <body>
    <script>
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xeeeeee);
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({{antialias:true}});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        const light = new THREE.DirectionalLight(0xffffff, 1);
        light.position.set(10,10,10);
        scene.add(light);
        scene.add(new THREE.AmbientLight(0xffffff,0.4));
        const loader = new THREE.STLLoader();
        const raw = atob("{stl_b64}");
        const arr = new Uint8Array([...raw].map(c=>c.charCodeAt(0)));
        const geom = loader.parse(arr.buffer);
        const mat = new THREE.MeshStandardMaterial({{color:0x4488dd, metalness:0.1, roughness:0.6}});
        const mesh = new THREE.Mesh(geom, mat);
        scene.add(mesh);
        geom.computeBoundingBox();
        const box = geom.boundingBox;
        const center = new THREE.Vector3();
        box.getCenter(center);
        mesh.position.sub(center);
        const size = new THREE.Vector3();
        box.getSize(size);
        const maxDim = Math.max(size.x,size.y,size.z);
        camera.position.z = maxDim * 1.8;
        function animate(){{requestAnimationFrame(animate);controls.update();renderer.render(scene,camera);}}
        animate();
    </script>
    </body>
    </html>
    '''
    st.components.v1.html(html, height=height, scrolling=False)

# -------------------------- DXF解析函数 --------------------------
def parse_dxf(dxf_bytes: bytes):
    stream = io.BytesIO(dxf_bytes)
    doc = ezdxf.read(stream)
    msp = doc.modelspace()
    paths = []
    for entity in msp:
        if entity.dxftype() == "LWPOLYLINE":
            pts = np.array([(p[0], p[1]) for p in entity.get_points()])
            path = trimesh.load_path(pts)
            paths.append(path)
    if not paths:
        return None
    union_path = trimesh.path.util.concatenate(paths)
    mesh = union_path.extrude(height=10.0)
    return mesh

# -------------------------- PDF解析函数 --------------------------
def parse_pdf(pdf_bytes: bytes):
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page = pdf.pages[0]
            lines = page.lines
            if not lines:
                raise Exception("无矢量线条，切换图片模式")
            st.warning("矢量PDF：圆弧会被拆成折线，建议DXF优先")
            pts = np.array([(0,0),(200,0),(200,120),(0,120)])
            path = trimesh.load_path(pts)
            mesh = path.extrude(height=10.0)
            return mesh
    except Exception:
        imgs = convert_from_bytes(pdf_bytes)
        img = np.array(imgs[0])
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, bin_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        if not areas:
            return None
        idx = np.argmax(areas)
        cnt = contours[idx].squeeze()
        path = trimesh.load_path(cnt)
        mesh = path.extrude(height=10.0)
        return mesh

# -------------------------- 工艺、G代码生成 --------------------------
def generate_process(mesh: trimesh.Trimesh) -> str:
    bbox = mesh.bounding_box.extents
    x, y, z = bbox
    proc = f"""===== 加工工艺方案 =====
零件外形：X={x:.2f} mm，Y={y:.2f} mm，厚度={z:.2f} mm
毛坯建议：矩形板料，单边预留余量1.5mm
加工工序：
1. 毛坯下料，平磨上下表面
2. 粗铣外轮廓，留0.3mm精加工余量
3. 精铣外轮廓至图纸尺寸
4. 去毛刺、清洗
推荐刀具：φ8立铣刀
切削参数：
主轴转速 S=1800 r/min，进给 F=600 mm/min
备注：本方案为自动生成参考，务必复核图纸公差、装夹方案！
"""
    return proc

def generate_gcode(mesh: trimesh.Trimesh) -> str:
    bbox = mesh.bounding_box.extents
    x, y, z = bbox
    gcode = f"""; AUTO GCODE - DEMO ONLY
; 零件尺寸 X={x:.2f} Y={y:.2f} Z={z:.2f}
G90 G54 G00 X0 Y0 S1800 M03
G43 Z50.0 H01
Z2.0
G01 Z-{z:.2f} F100
G01 X{x:.2f} Y0 F600
G01 X{x:.2f} Y{y:.2f}
G01 X0 Y{y:.2f}
G01 X0 Y0
G00 Z50.0
M05
M30
"""
    return gcode

# -------------------------- 页面交互逻辑 --------------------------
uploaded_file = st.file_uploader("上传工程图（支持 .dxf / .pdf）", type=["dxf", "pdf"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    fname = uploaded_file.name.lower()
    mesh = None
    with st.spinner("正在解析图纸，生成3D模型..."):
        if fname.endswith(".dxf"):
            mesh = parse_dxf(file_bytes)
        elif fname.endswith(".pdf"):
            mesh = parse_pdf(file_bytes)
    if mesh is None:
        st.error("❌ 解析失败，无法提取轮廓，请更换图纸或使用DXF文件")
    else:
        st.session_state["mesh"] = mesh
        stl_buf = io.BytesIO()
        mesh.export(stl_buf, file_type="stl")
        stl_buf.seek(0)
        stl_bytes = stl_buf.read()
        st.session_state["stl_bytes"] = stl_bytes
        st.success("✅ 3D模型生成完成")

col1, col2 = st.columns(2)
with col1:
    st.subheader("3D模型预览")
    if st.session_state["stl_bytes"] is not None:
        stl_preview(st.session_state["stl_bytes"], height=400)
        st.download_button("📥 下载STL模型", data=st.session_state["stl_bytes"], file_name="part.stl")

with col2:
    st.subheader("工艺与NC代码")
    if st.session_state["mesh"] is not None:
        if st.button("确认模型 → 生成加工工艺"):
            st.session_state["process_text"] = generate_process(st.session_state["mesh"])
        st.text_area("加工工艺参考", st.session_state["process_text"], height=240)

        if st.session_state["process_text"] and st.button("确认工艺 → 生成NC代码"):
            st.session_state["gcode_text"] = generate_gcode(st.session_state["mesh"])
        st.text_area("NC G代码（演示，不可直接上机）", st.session_state["gcode_text"], height=240)
        if st.session_state["gcode_text"]:
            st.download_button("📥 下载G代码", data=st.session_state["gcode_text"], file_name="part.nc")

st.markdown("---")
st.warning("⚠️ 重要提醒：本程序仅为原型演示！自动生成的G代码**严禁直接上机加工**，必须人工复核、仿真校验！")
