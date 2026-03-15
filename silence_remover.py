import os
import io
import math
import array
import zipfile
import threading
from flask import Flask, request, jsonify, send_file, render_template_string
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from werkzeug.utils import secure_filename
import uuid

# ── 설정 ────────────────────────────────────────────────
UPLOAD_DIR  = "/home/claude/uploads"
OUTPUT_DIR  = "/home/claude/outputs"
SILENCE_THRESH_OFFSET = -16
MIN_SILENCE_LEN = 100

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB

# ── RMS 계산 ─────────────────────────────────────────────
def rms_dbfs(audio: AudioSegment) -> float:
    """구간의 RMS 에너지를 dBFS로 반환"""
    samples = audio.get_array_of_samples()
    if not samples:
        return -float("inf")
    max_val = float(2 ** (audio.sample_width * 8 - 1))
    rms = math.sqrt(sum(s * s for s in samples) / len(samples))
    if rms == 0:
        return -float("inf")
    return 20 * math.log10(rms / max_val)


def calc_rms_thresh(audio: AudioSegment, thresh_offset: int) -> float:
    """최대 RMS 기준으로 임계값 계산"""
    win = 20
    max_rms = -float("inf")
    for i in range(0, len(audio) - win, win):
        db = rms_dbfs(audio[i:i+win])
        if db > max_rms:
            max_rms = db
    return max_rms + thresh_offset


# ── 핵심 로직 ────────────────────────────────────────────
def detect_mode(audio: AudioSegment) -> str:
    return "long" if len(audio) >= 60_000 else "short"


def find_boundaries_rms(audio: AudioSegment, thresh_offset: int, min_silence_len: int) -> tuple:
    """
    RMS 에너지 기반으로 앞뒤 무음 경계를 1ms 단위로 탐색.
    페이드인/아웃 구간도 정확히 감지.
    """
    thresh = calc_rms_thresh(audio, thresh_offset)
    length = len(audio)
    win    = 20  # 20ms 윈도우로 RMS 측정

    # 앞부분: 앞에서부터 RMS 임계값 넘는 첫 지점
    start = 0
    for i in range(0, length - win, 1):
        if rms_dbfs(audio[i:i+win]) > thresh:
            start = i
            break

    # 뒷부분: 뒤에서부터 RMS 임계값 넘는 첫 지점
    end = length
    for i in range(length - win, 0, -1):
        if rms_dbfs(audio[i:i+win]) > thresh:
            end = i + win
            break

    return start, end


def find_nonsilent_rms(audio: AudioSegment, thresh_offset: int, min_silence_len: int) -> list:
    """
    RMS 기반으로 소리 있는 구간 목록 반환 (long 모드용).
    """
    thresh = calc_rms_thresh(audio, thresh_offset)
    length = len(audio)
    win    = 20

    is_sound = []
    for i in range(0, length - win, win):
        db = rms_dbfs(audio[i:i+win])
        is_sound.append((i, db > thresh))

    # 연속 구간 묶기
    nonsilent = []
    in_sound  = False
    seg_start = 0
    for i, (ms, sound) in enumerate(is_sound):
        if sound and not in_sound:
            seg_start = ms
            in_sound  = True
        elif not sound and in_sound:
            # min_silence_len 이상 무음이면 구간 종료
            silent_ms = is_sound[i][0] - ms
            if ms - seg_start >= min_silence_len:
                nonsilent.append([seg_start, ms])
                in_sound = False
    if in_sound:
        nonsilent.append([seg_start, length])

    return nonsilent


def remove_silence(audio: AudioSegment, mode: str, thresh_offset: int = SILENCE_THRESH_OFFSET, min_silence_len: int = MIN_SILENCE_LEN) -> AudioSegment:
    if mode == "short":
        start, end = find_boundaries_rms(audio, thresh_offset, min_silence_len)
        if start >= end:
            return audio
        return audio[start:end]
    else:
        nonsilent = find_nonsilent_rms(audio, thresh_offset, min_silence_len)
        if not nonsilent:
            return audio
        result = audio[nonsilent[0][0]:nonsilent[0][1]]
        for s, e in nonsilent[1:]:
            result = result + audio[s:e]
        return result

def process_file(src: str, dst: str, thresh_offset: int = -40, min_silence_len: int = 100) -> dict:
    audio   = AudioSegment.from_file(src)
    mode    = detect_mode(audio)
    trimmed = remove_silence(audio, mode, thresh_offset, min_silence_len)
    trimmed.export(dst, format="mp3", bitrate="192k")
    removed_ms = len(audio) - len(trimmed)
    return {
        "mode":        "긴 파일" if mode == "long" else "문장",
        "original_s":  round(len(audio) / 1000, 2),
        "trimmed_s":   round(len(trimmed) / 1000, 2),
        "removed_s":   round(removed_ms / 1000, 2),
        "removed_pct": round(removed_ms / len(audio) * 100, 1) if len(audio) else 0,
    }

# ── ZIP에서 mp3 추출 ─────────────────────────────────────
def extract_mp3_from_zip(zip_path: str) -> list[tuple[str, str]]:
    """ZIP 안의 mp3 파일을 UPLOAD_DIR에 풀고 (원본명, 경로) 리스트 반환"""
    extracted = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            # __MACOSX 등 숨김 폴더 무시, mp3만 처리
            if info.filename.startswith("__") or info.is_dir():
                continue
            name_lower = info.filename.lower()
            if not name_lower.endswith(".mp3"):
                continue
            orig_name = os.path.basename(info.filename)
            safe_name = secure_filename(orig_name) or f"{uuid.uuid4()}.mp3"
            dst = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{safe_name}")
            with zf.open(info) as src, open(dst, "wb") as out:
                out.write(src.read())
            extracted.append((orig_name, dst))
    return extracted


# ── API 라우트 ───────────────────────────────────────────
@app.route("/api/process", methods=["POST"])
def api_process():
    uploads = request.files.getlist("files")
    if not uploads:
        return jsonify({"error": "파일이 없습니다"}), 400

    thresh_offset   = int(request.form.get("thresh_offset", -40))
    min_silence_len = int(request.form.get("min_silence_len", 100))

    # (원본 파일명, 임시 저장 경로) 목록 구성
    job_list: list[tuple[str, str]] = []
    zip_temps: list[str] = []

    for f in uploads:
        orig_name = f.filename or "unknown"
        safe_name = secure_filename(orig_name) or f"{uuid.uuid4()}.mp3"
        tmp_path  = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{safe_name}")
        f.save(tmp_path)

        if orig_name.lower().endswith(".zip"):
            # ZIP → 내부 mp3 추출
            zip_temps.append(tmp_path)
            try:
                extracted = extract_mp3_from_zip(tmp_path)
                job_list.extend(extracted)
            except Exception as e:
                job_list.append((orig_name, None))  # 오류 표시용
        else:
            job_list.append((orig_name, tmp_path))

    results = []
    for orig_name, src_path in job_list:
        if src_path is None:
            results.append({"original_name": orig_name, "status": "error", "error": "ZIP 읽기 실패"})
            continue

        name, ext  = os.path.splitext(secure_filename(orig_name))
        out_name   = f"{name}_trimmed.mp3"
        dst_path   = os.path.join(OUTPUT_DIR, out_name)

        try:
            stats = process_file(src_path, dst_path, thresh_offset, min_silence_len)
            results.append({
                "original_name": orig_name,
                "output_name":   out_name,
                "status":        "success",
                **stats,
            })
        except Exception as e:
            results.append({"original_name": orig_name, "status": "error", "error": str(e)})
        finally:
            if os.path.exists(src_path):
                os.remove(src_path)

    # ZIP 임시파일 정리
    for z in zip_temps:
        if os.path.exists(z):
            os.remove(z)

    return jsonify(results)


@app.route("/api/download/<filename>")
def api_download(filename):
    path = os.path.join(OUTPUT_DIR, secure_filename(filename))
    if not os.path.exists(path):
        return jsonify({"error": "파일 없음"}), 404
    return send_file(path, as_attachment=True)


@app.route("/api/download-zip", methods=["POST"])
def api_download_zip():
    data      = request.get_json()
    filenames = data.get("filenames", [])
    if not filenames:
        return jsonify({"error": "파일 없음"}), 400

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in filenames:
            path = os.path.join(OUTPUT_DIR, secure_filename(name))
            if os.path.exists(path):
                zf.write(path, arcname=name)
    buf.seek(0)
    return send_file(buf, mimetype="application/zip",
                     as_attachment=True, download_name="trimmed_output.zip")


@app.route("/")
def index():
    return render_template_string(HTML)


# ── HTML UI ──────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TTS 무음 제거기</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0f0f1a;
    color: #cdd6f4;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 40px 16px;
  }
  h1 { font-size: 28px; font-weight: 700; margin-bottom: 6px; }
  .sub { color: #6c7086; font-size: 14px; margin-bottom: 32px; }

  /* 드롭존 */
  #dropzone {
    width: 100%;
    max-width: 700px;
    border: 2px dashed #45475a;
    border-radius: 16px;
    padding: 48px 24px;
    text-align: center;
    cursor: pointer;
    transition: border-color .2s, background .2s;
  }
  #dropzone.over { border-color: #89b4fa; background: rgba(137,180,250,.06); }
  #dropzone svg { width: 40px; height: 40px; margin-bottom: 12px; opacity: .5; }
  #dropzone p { color: #a6adc8; font-size: 14px; }
  #dropzone span { color: #89b4fa; }
  #file-input { display: none; }

  /* 파일 목록 */
  #file-list {
    width: 100%;
    max-width: 700px;
    margin-top: 20px;
    display: none;
    flex-direction: column;
    gap: 8px;
  }
  .file-row {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 10px;
    padding: 12px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .file-row .name { flex: 1; font-size: 13px; color: #cdd6f4; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .file-row .dur  { font-size: 12px; color: #6c7086; min-width: 48px; text-align: right; }
  .file-row .status { font-size: 12px; min-width: 120px; text-align: right; }
  .status.pending  { color: #6c7086; }
  .status.done     { color: #a6e3a1; }
  .status.error    { color: #f38ba8; }
  .status.running  { color: #89b4fa; }
  .dl-btn {
    background: #313244; border: none; border-radius: 6px;
    color: #89b4fa; font-size: 11px; padding: 4px 10px;
    cursor: pointer; display: none;
  }
  .dl-btn:hover { background: #45475a; }

  /* 버튼 */
  .btn-row { display: flex; gap: 12px; margin-top: 20px; width: 100%; max-width: 700px; }
  button.primary {
    flex: 1; padding: 14px; border: none; border-radius: 10px;
    background: #89b4fa; color: #1e1e2e; font-size: 15px; font-weight: 700;
    cursor: pointer; transition: opacity .2s;
  }
  button.primary:disabled { opacity: .4; cursor: default; }
  button.secondary {
    padding: 14px 20px; border: none; border-radius: 10px;
    background: #313244; color: #a6adc8; font-size: 14px;
    cursor: pointer; transition: background .2s;
  }
  button.secondary:hover { background: #45475a; }

  /* 진행 바 */
  #progress-wrap { width: 100%; max-width: 700px; margin-top: 16px; display: none; }
  #progress-bar  {
    height: 6px; background: #313244; border-radius: 3px; overflow: hidden;
  }
  #progress-fill { height: 100%; background: #89b4fa; width: 0%; transition: width .3s; border-radius: 3px; }
  #progress-label { font-size: 12px; color: #6c7086; margin-top: 6px; text-align: center; }
</style>
</head>
<body>
<h1>🎙 TTS 무음 제거기</h1>
<p class="sub">mp3 파일의 앞·뒤·중간 무음을 자동으로 제거합니다</p>

<div id="dropzone" onclick="document.getElementById('file-input').click()">
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
    <path d="M12 16V4m0 0L8 8m4-4 4 4M4 20h16" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>
  <p><span>클릭</span>하거나 파일을 여기에 드래그하세요</p>
  <p style="margin-top:6px;font-size:12px;">MP3 파일 여러 개 · ZIP 파일 동시 업로드 가능</p>
</div>
<input type="file" id="file-input" accept=".mp3,.zip,audio/*" multiple>

<div id="file-list"></div>

<!-- 설정 패널 -->
<div style="width:100%;max-width:700px;background:#1e1e2e;border:1px solid #313244;border-radius:12px;padding:20px;margin-top:20px;">
  <div style="font-size:13px;font-weight:700;color:#a6adc8;margin-bottom:16px;">⚙️ 무음 감지 설정</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">

    <!-- 무음 임계값 -->
    <div>
      <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
        <label style="font-size:12px;color:#a6adc8;">무음 임계값 (dB 오프셋)</label>
        <span id="thresh-label" style="font-size:12px;font-weight:700;color:#89b4fa;">-16 dB</span>
      </div>
      <input type="range" id="thresh-slider" min="-40" max="-4" value="-16" step="1"
        style="width:100%;accent-color:#89b4fa;"
        oninput="document.getElementById('thresh-label').textContent = this.value + ' dB'; document.getElementById('thresh-input').value = this.value">
      <div style="display:flex;justify-content:space-between;margin-top:4px;">
        <span style="font-size:10px;color:#45475a;">-40 dB (약하게)</span>
        <span style="font-size:10px;color:#45475a;">-4 dB (강하게)</span>
      </div>
      <div style="margin-top:10px;display:flex;align-items:center;gap:8px;">
        <span style="font-size:12px;color:#6c7086;">직접 입력:</span>
        <input type="number" id="thresh-input" value="-16" min="-40" max="-4"
          style="width:80px;background:#313244;border:1px solid #45475a;border-radius:6px;color:#cdd6f4;padding:4px 8px;font-size:12px;"
          oninput="syncSlider(this.value)">
        <span style="font-size:12px;color:#6c7086;">dB</span>
      </div>
      <div style="margin-top:8px;font-size:11px;color:#585b70;">
        값이 클수록(0에 가까울수록) 작은 소리도 제거
      </div>
    </div>

    <!-- 최소 무음 길이 -->
    <div>
      <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
        <label style="font-size:12px;color:#a6adc8;">최소 무음 길이</label>
        <span id="msl-label" style="font-size:12px;font-weight:700;color:#89b4fa;">100 ms</span>
      </div>
      <input type="range" id="msl-slider" min="10" max="1000" value="100" step="10"
        style="width:100%;accent-color:#89b4fa;"
        oninput="document.getElementById('msl-label').textContent = this.value + ' ms'; document.getElementById('msl-input').value = this.value">
      <div style="display:flex;justify-content:space-between;margin-top:4px;">
        <span style="font-size:10px;color:#45475a;">10ms (민감)</span>
        <span style="font-size:10px;color:#45475a;">1000ms (둔감)</span>
      </div>
      <div style="margin-top:10px;display:flex;align-items:center;gap:8px;">
        <span style="font-size:12px;color:#6c7086;">직접 입력:</span>
        <input type="number" id="msl-input" value="100" min="10" max="1000"
          style="width:80px;background:#313244;border:1px solid #45475a;border-radius:6px;color:#cdd6f4;padding:4px 8px;font-size:12px;"
          oninput="syncMsl(this.value)">
        <span style="font-size:12px;color:#6c7086;">ms</span>
      </div>
      <div style="margin-top:8px;font-size:11px;color:#585b70;">
        이 길이 이상의 무음만 제거됨
      </div>
    </div>

  </div>

  <!-- 프리셋 버튼 -->
  <div style="margin-top:16px;display:flex;gap:8px;align-items:center;">
    <span style="font-size:12px;color:#6c7086;">프리셋:</span>
    <button onclick="setPreset(-16,100)" style="background:#313244;border:none;border-radius:6px;color:#a6adc8;font-size:11px;padding:5px 12px;cursor:pointer;">기본값</button>
    <button onclick="setPreset(-12,80)"  style="background:#313244;border:none;border-radius:6px;color:#a6adc8;font-size:11px;padding:5px 12px;cursor:pointer;">작은소리 제거</button>
    <button onclick="setPreset(-8,50)"   style="background:#313244;border:none;border-radius:6px;color:#f38ba8;font-size:11px;padding:5px 12px;cursor:pointer;">강하게 제거</button>
  </div>
</div>

<div class="btn-row">
  <button class="primary" id="run-btn" disabled onclick="runProcess()">무음 제거 시작</button>
  <button class="secondary" id="zip-btn" style="display:none;" onclick="downloadZip()">📦 ZIP 다운로드</button>
  <button class="secondary" onclick="clearAll()">목록 지우기</button>
</div>

<div id="progress-wrap">
  <div id="progress-bar"><div id="progress-fill"></div></div>
  <p id="progress-label">0 / 0</p>
</div>

<script>
function syncSlider(v) {
  v = Math.max(-40, Math.min(-4, parseInt(v) || -16));
  document.getElementById('thresh-slider').value = v;
  document.getElementById('thresh-label').textContent = v + ' dB';
  document.getElementById('thresh-input').value = v;
}
function syncMsl(v) {
  v = Math.max(10, Math.min(1000, parseInt(v) || 100));
  document.getElementById('msl-slider').value = v;
  document.getElementById('msl-label').textContent = v + ' ms';
  document.getElementById('msl-input').value = v;
}
function setPreset(thresh, msl) {
  syncSlider(thresh);
  syncMsl(msl);
}

let files = [];
let completedFiles = [];  // 처리 완료된 파일명 저장

// 드래그앤드롭
const dz = document.getElementById('dropzone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('over'); });
dz.addEventListener('dragleave', () => dz.classList.remove('over'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('over');
  addFiles([...e.dataTransfer.files].filter(f =>
    f.type.includes('audio') || f.name.endsWith('.mp3') || f.name.endsWith('.zip')
  ));
});
document.getElementById('file-input').addEventListener('change', e => {
  addFiles([...e.target.files]);
  e.target.value = '';
});

function addFiles(newFiles) {
  newFiles.forEach(f => {
    if (!files.find(x => x.name === f.name && x.size === f.size)) {
      files.push(f);
      renderRow(f, files.length - 1);
    }
  });
  document.getElementById('file-list').style.display = files.length ? 'flex' : 'none';
  document.getElementById('run-btn').disabled = files.length === 0;
}

function renderRow(f, idx) {
  const list   = document.getElementById('file-list');
  const row    = document.createElement('div');
  row.className = 'file-row';
  row.id = `row-${idx}`;
  const sizeMB = (f.size / 1024 / 1024).toFixed(1);
  const icon   = f.name.endsWith('.zip') ? '📦' : '🎵';
  const tag    = f.name.endsWith('.zip')
    ? `<span style="font-size:10px;background:#45475a;color:#cba6f7;border-radius:4px;padding:2px 6px;">ZIP</span>`
    : `<span style="font-size:10px;background:#45475a;color:#89b4fa;border-radius:4px;padding:2px 6px;">MP3</span>`;
  row.innerHTML = `
    <span style="font-size:16px;">${icon}</span>
    <div class="name" title="${f.name}">${f.name}</div>
    ${tag}
    <div class="dur">${sizeMB}MB</div>
    <div class="status pending" id="st-${idx}">대기 중</div>
    <button class="dl-btn" id="dl-${idx}">다운로드</button>
  `;
  list.appendChild(row);
}

function clearAll() {
  files = [];
  completedFiles = [];
  document.getElementById('file-list').innerHTML = '';
  document.getElementById('file-list').style.display = 'none';
  document.getElementById('run-btn').disabled = true;
  document.getElementById('zip-btn').style.display = 'none';
  document.getElementById('progress-wrap').style.display = 'none';
}

async function runProcess() {
  if (!files.length) return;
  document.getElementById('run-btn').disabled = true;
  document.getElementById('zip-btn').style.display = 'none';
  document.getElementById('progress-wrap').style.display = 'block';
  completedFiles = [];

  const total = files.length;
  let done = 0;

  for (let i = 0; i < files.length; i++) {
    const st = document.getElementById(`st-${i}`);
    st.className = 'status running';
    st.textContent = '처리 중...';

    const fd = new FormData();
    fd.append('files', files[i]);
    fd.append('thresh_offset',   document.getElementById('thresh-input').value);
    fd.append('min_silence_len', document.getElementById('msl-input').value);

    try {
      const res  = await fetch('/api/process', { method: 'POST', body: fd });
      const data = await res.json();

      if (files[i].name.endsWith('.zip')) {
        // ZIP → 여러 결과
        const ok    = data.filter(r => r.status === 'success');
        const fails = data.filter(r => r.status !== 'success');
        ok.forEach(r => completedFiles.push(r.output_name));
        if (fails.length === 0) {
          st.className = 'status done';
          st.textContent = `ZIP 내 ${ok.length}개 완료`;
        } else {
          st.className = 'status error';
          st.textContent = `${ok.length}개 완료 / ${fails.length}개 오류`;
        }
        // ZIP 결과는 ZIP 다운로드로만 제공
        const dlBtn = document.getElementById(`dl-${i}`);
        if (ok.length > 0) {
          dlBtn.style.display = 'inline-block';
          dlBtn.textContent = `ZIP (${ok.length}개)`;
          dlBtn.onclick = () => downloadSpecificZip(ok.map(r => r.output_name));
        }
      } else {
        // 단일 mp3
        const r = data[0];
        if (r.status === 'success') {
          st.className = 'status done';
          st.textContent = `-${r.removed_pct}% (${r.removed_s}s 제거)`;
          completedFiles.push(r.output_name);
          const dlBtn = document.getElementById(`dl-${i}`);
          dlBtn.style.display = 'inline-block';
          dlBtn.textContent = '다운로드';
          dlBtn.onclick = () => window.location.href = `/api/download/${encodeURIComponent(r.output_name)}`;
        } else {
          st.className = 'status error';
          st.textContent = '오류: ' + r.error;
        }
      }
    } catch(e) {
      st.className = 'status error';
      st.textContent = '네트워크 오류';
    }

    done++;
    document.getElementById('progress-fill').style.width = `${done/total*100}%`;
    document.getElementById('progress-label').textContent = `${done} / ${total} 완료`;
  }

  document.getElementById('run-btn').disabled = false;
  if (completedFiles.length > 1) {
    document.getElementById('zip-btn').style.display = 'inline-block';
  }
}

async function downloadSpecificZip(filenames) {
  try {
    const res = await fetch('/api/download-zip', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filenames })
    });
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'trimmed_output.zip';
    a.click();
    URL.revokeObjectURL(url);
  } catch(e) { alert('다운로드 실패: ' + e.message); }
}

async function downloadZip() {
  if (!completedFiles.length) return;
  const btn = document.getElementById('zip-btn');
  btn.textContent = '⏳ 압축 중...';
  btn.disabled = true;
  try {
    const res = await fetch('/api/download-zip', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filenames: completedFiles })
    });
    if (!res.ok) throw new Error('서버 오류');
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'trimmed_output.zip';
    a.click();
    URL.revokeObjectURL(url);
  } catch(e) {
    alert('ZIP 다운로드 실패: ' + e.message);
  }
  btn.textContent = '📦 ZIP 다운로드';
  btn.disabled = false;
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
