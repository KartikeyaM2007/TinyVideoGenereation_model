# run_final_app_v2.ps1
# Clean launcher for AnomVidGen final Streamlit app

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== AnomVidGen Final App Launcher v2 ===" -ForegroundColor Cyan

# Always run from this script folder
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Activate venv if present
$venvActivate = Join-Path $root ".venv\Scripts\Activate.ps1"
if (Test-Path -LiteralPath $venvActivate) {
    Write-Host "[ok] Activating .venv" -ForegroundColor Green
    . $venvActivate
} else {
    Write-Host "[warn] .venv not found. Using current Python environment." -ForegroundColor Yellow
}

# Create output folders
New-Item -ItemType Directory -Force -Path "outputs\app" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\video2f_samples" | Out-Null

# Create final Streamlit app using line array, no here-strings
$appLines = @(
'from pathlib import Path',
'import subprocess',
'import sys',
'import streamlit as st',
'',
'CLASS_NAMES = ["Normal", "Fighting", "RoadAccidents"]',
'',
'PROMPT_MAP = {',
'    "normal": "Normal",',
'    "safe": "Normal",',
'    "street": "Normal",',
'    "surveillance": "Normal",',
'    "fight": "Fighting",',
'    "fighting": "Fighting",',
'    "violence": "Fighting",',
'    "violent": "Fighting",',
'    "assault": "Fighting",',
'    "accident": "RoadAccidents",',
'    "crash": "RoadAccidents",',
'    "road": "RoadAccidents",',
'    "traffic": "RoadAccidents",',
'    "vehicle": "RoadAccidents",',
'    "collision": "RoadAccidents",',
'}',
'',
'def prompt_to_class(prompt: str) -> str:',
'    text = prompt.lower()',
'    for keyword, class_name in PROMPT_MAP.items():',
'        if keyword in text:',
'            return class_name',
'    return "Normal"',
'',
'def run_command(cmd):',
'    result = subprocess.run(cmd, text=True, capture_output=True, shell=False)',
'    if result.returncode != 0:',
'        st.error("Command failed")',
'        st.code(result.stderr or result.stdout)',
'        return False',
'    if result.stdout:',
'        with st.expander("Terminal output"):',
'            st.code(result.stdout)',
'    return True',
'',
'def generate_image(class_name: str, steps: int, guidance: float):',
'    out_dir = Path("outputs/app")',
'    out_dir.mkdir(parents=True, exist_ok=True)',
'    output_path = out_dir / f"{class_name}_image.png"',
'    cmd = [',
'        sys.executable,',
'        "sample_image_diffusion.py",',
'        "--config", "configs/image2d_class.yaml",',
'        "--class_name", class_name,',
'        "--output", str(output_path),',
'        "--steps", str(steps),',
'        "--guidance", str(guidance),',
'    ]',
'    ok = run_command(cmd)',
'    return ok, output_path',
'',
'def generate_2frame(class_name: str, steps: int, guidance: float):',
'    cmd = [',
'        sys.executable,',
'        "sample_2frame_diffusion.py",',
'        "--config", "configs/video2f_class.yaml",',
'        "--class_name", class_name,',
'        "--steps", str(steps),',
'        "--guidance", str(guidance),',
'    ]',
'    ok = run_command(cmd)',
'    sample_dir = Path("outputs/video2f_samples")',
'    gif_candidates = sorted(sample_dir.glob(f"{class_name}*.gif"), key=lambda p: p.stat().st_mtime, reverse=True)',
'    png_candidates = sorted(sample_dir.glob(f"{class_name}*.png"), key=lambda p: p.stat().st_mtime, reverse=True)',
'    gif_path = gif_candidates[0] if gif_candidates else sample_dir / f"{class_name}_sample_1.gif"',
'    side_path = png_candidates[0] if png_candidates else sample_dir / f"{class_name}_sample_1_sidebyside.png"',
'    return ok, gif_path, side_path',
'',
'def main():',
'    st.set_page_config(page_title="AnomVidGen", page_icon="🎥", layout="wide")',
'    st.title("🎥 AnomVidGen")',
'    st.caption("Low VRAM anomaly image and 2 frame video generation using compact PyTorch diffusion models.")',
'',
'    with st.sidebar:',
'        st.header("Generation Settings")',
'        mode = st.radio("Mode", ["Image generation", "2 frame GIF generation"])',
'        input_type = st.radio("Input type", ["Class selector", "Text prompt"])',
'        if input_type == "Class selector":',
'            class_name = st.selectbox("Class", CLASS_NAMES)',
'        else:',
'            prompt = st.text_input("Prompt", value="a road accident surveillance scene")',
'            class_name = prompt_to_class(prompt)',
'            st.info(f"Mapped prompt to class: {class_name}")',
'        steps = st.slider("Sampling steps", min_value=50, max_value=500, value=100, step=50)',
'        guidance = st.slider("Guidance scale", min_value=1.0, max_value=3.0, value=1.0, step=0.5)',
'        generate = st.button("Generate", type="primary")',
'',
'    st.subheader("Final Pipeline")',
'    st.code("class/text input -> class conditioned diffusion -> surveillance image or 2 frame GIF", language="text")',
'',
'    if generate:',
'        with st.spinner("Generating..."):',
'            if mode == "Image generation":',
'                ok, output_path = generate_image(class_name, steps, guidance)',
'                if ok and output_path.exists():',
'                    st.success(f"Generated image for {class_name}")',
'                    st.image(str(output_path), caption=str(output_path))',
'                else:',
'                    st.warning("Image output was not found.")',
'            else:',
'                ok, gif_path, side_path = generate_2frame(class_name, steps, guidance)',
'                if ok:',
'                    col1, col2 = st.columns(2)',
'                    if gif_path.exists():',
'                        col1.image(str(gif_path), caption="Generated 2 frame GIF")',
'                    else:',
'                        col1.warning(f"Missing GIF: {gif_path}")',
'                    if side_path.exists():',
'                        col2.image(str(side_path), caption="Side by side frames")',
'                    else:',
'                        col2.warning(f"Missing side by side image: {side_path}")',
'',
'    st.markdown("---")',
'    st.subheader("Model Notes")',
'    st.markdown("""',
'    This is the final stable version of the project.',
'',
'    **Final decision**',
'    - Use class conditioned image diffusion',
'    - Use class conditioned 2 frame diffusion',
'    - Keep 4 frame and transition models as experiments',
'',
'    **Limitations**',
'    - Low resolution outputs',
'    - Blurry generation',
'    - Not a general Sora style text to video model',
'    - Text prompts are mapped to fixed anomaly classes',
'    """)',
'',
'if __name__ == "__main__":',
'    main()'
)
Set-Content -Path "app_final.py" -Value $appLines -Encoding UTF8
Write-Host "[ok] app_final.py created" -ForegroundColor Green

# Required files, built with Join-Path so no bad path characters can break Test-Path
$requiredFiles = @(
    (Join-Path $root "configs\image2d_class.yaml"),
    (Join-Path $root "configs\video2f_class.yaml"),
    (Join-Path $root "sample_image_diffusion.py"),
    (Join-Path $root "sample_2frame_diffusion.py"),
    (Join-Path $root "runs\image2d_class\latest.pt"),
    (Join-Path $root "runs\video2f_class\latest.pt")
)

foreach ($file in $requiredFiles) {
    $cleanPath = [string]$file
    if (!(Test-Path -LiteralPath $cleanPath)) {
        Write-Host "[missing] $cleanPath" -ForegroundColor Red
        Write-Host "Fix this missing file/checkpoint before running the final app." -ForegroundColor Red
        exit 1
    }
}
Write-Host "[ok] Required files found" -ForegroundColor Green

# Check Streamlit
python -c "import streamlit" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[info] Installing streamlit..." -ForegroundColor Yellow
    pip install streamlit
}

Write-Host ""
Write-Host "Starting final app..." -ForegroundColor Cyan
streamlit run app_final.py
