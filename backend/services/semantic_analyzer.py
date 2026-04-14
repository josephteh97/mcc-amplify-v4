"""
Semantic Analysis - Multi-Backend AI Analyzer
Supports: Google Gemini API, Anthropic Claude API, Local LLaVA, Local Qwen2.5-VL, Ollama (vision models)

Configure via environment variable SEMANTIC_MODEL_BACKEND:
  gemini_api   - Google Gemini via API (default)
  anthropic_api- Anthropic Claude via API
  local_llava  - Local LLaVA model at LOCAL_MODELS_DIR/llava (GPU required)
  local_qwen   - Local Qwen2.5-VL model at LOCAL_MODELS_DIR/qwen (GPU required)

Backend priority is controlled by the environment variable SEMANTIC_BACKEND_PRIORITY.
Example: SEMANTIC_BACKEND_PRIORITY="ollama,gemini_api" tries Ollama first, then Gemini.
If not set, it falls back to the single backend defined in SEMANTIC_MODEL_BACKEND.
"""

import os
import json
import time
import threading
from typing import Dict, List, Optional
from pathlib import Path
from backend.services.geometry_generator import (
    STANDARD_SQUARE_COLUMN_SIZES,
    STANDARD_CIRCULAR_COLUMN_DIAMETERS,
)
from dotenv import load_dotenv
from loguru import logger
from PIL import Image

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# Backend priority list (comma-separated, e.g., "ollama,gemini_api")
BACKEND_PRIORITY = [
    b.strip() for b in os.getenv("SEMANTIC_BACKEND_PRIORITY", "").split(",") if b.strip()
] or [os.getenv("SEMANTIC_MODEL_BACKEND", "gemini_api")]

LOCAL_MODELS_DIR = Path(os.getenv("LOCAL_MODELS_DIR", "../models"))

# Ollama settings
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# ── Gemini free-tier rate limiter ─────────────────────────────────────────────
_GEMINI_LOCK          = threading.Lock()
_GEMINI_LAST_CALL_TS  = 0.0
_GEMINI_MIN_INTERVAL  = float(os.getenv("GEMINI_MIN_INTERVAL_S", "13"))


class SemanticAnalyzer:
    """
    Multi-backend semantic analyzer for floor plan understanding.

    The active backend is determined by trying the backends listed in
    SEMANTIC_BACKEND_PRIORITY (or SEMANTIC_MODEL_BACKEND) in order.
    The first successfully initialised backend is used for all calls.
    """

    def __init__(self):
        self.backend = None
        self.client = None
        self.model_id = None

        # Try each backend in priority order
        for backend_name in BACKEND_PRIORITY:
            logger.info(f"Attempting to initialise backend: {backend_name}")
            try:
                self._init_backend(backend_name)
                self.backend = backend_name
                logger.info(f"✓ Successfully initialised {backend_name}")
                break
            except Exception as e:
                logger.warning(f"Backend {backend_name} failed: {e}")
        else:
            logger.warning(
                f"No semantic AI backend could be initialised. Tried: {BACKEND_PRIORITY}. "
                "Add GOOGLE_API_KEY or ANTHROPIC_API_KEY to backend/.env — "
                "semantic analysis will be skipped and YOLO detections used as-is."
            )

    def _init_backend(self, backend_name: str):
        """Initialise a specific backend by name."""
        if backend_name == "gemini_api":
            self._init_gemini()
        elif backend_name == "anthropic_api":
            self._init_anthropic()
        elif backend_name == "local_llava":
            self._init_local_llava()
        elif backend_name == "local_qwen":
            self._init_local_qwen()
        elif backend_name == "ollama":
            self._init_ollama()
        else:
            raise ValueError(f"Unknown backend: {backend_name}")

    # ------------------------------------------------------------------
    # Gemini initializer
    # ------------------------------------------------------------------
    def _init_gemini(self):
        """Initialize Google Gemini API client."""
        from google import genai
        from google.genai import types as genai_types

        from utils.api_keys import get_google_api_key
        api_key = get_google_api_key()
        if not api_key or api_key == "[placeholder]":
            raise ValueError(
                "GOOGLE_API_KEY not set. Add it to backend/.env or backend/google_key.txt."
            )

        self._genai_types = genai_types
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.5-flash"

    # ------------------------------------------------------------------
    # Anthropic initializer
    # ------------------------------------------------------------------
    def _init_anthropic(self):
        """Initialize Anthropic Claude API client."""
        import anthropic
        from utils.api_keys import get_anthropic_api_key

        api_key = get_anthropic_api_key()
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. Add it to backend/.env (ANTHROPIC_API_KEY) "
                "or backend/claude_key.txt."
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_id = "claude-opus-4-6"

    # ------------------------------------------------------------------
    # Local LLaVA (placeholder)
    # ------------------------------------------------------------------
    def _init_local_llava(self):
        """Initialize local LLaVA model (placeholder)."""
        model_path = LOCAL_MODELS_DIR / "llava"
        if not model_path.exists():
            raise FileNotFoundError(
                f"LLaVA model directory not found: {model_path}\n"
                f"Download the model and place it at {model_path}"
            )
        # Placeholder – uncomment when ready
        raise NotImplementedError(
            "Local LLaVA inference is not yet activated. "
            "Uncomment the model loading code in _init_local_llava() when GPU is ready."
        )

    # ------------------------------------------------------------------
    # Local Qwen2.5-VL (placeholder)
    # ------------------------------------------------------------------
    def _init_local_qwen(self):
        """Initialize local Qwen2.5-VL model (placeholder)."""
        model_path = LOCAL_MODELS_DIR / "qwen"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Qwen model directory not found: {model_path}\n"
                f"Download the model and place it at {model_path}"
            )
        # Placeholder – uncomment when ready
        raise NotImplementedError(
            "Local Qwen2.5-VL inference is not yet activated. "
            "Uncomment the model loading code in _init_local_qwen() when GPU is ready."
        )

    # ------------------------------------------------------------------
    # Ollama (vision models) initializer with auto-start
    # ------------------------------------------------------------------
    def _init_ollama(self):
        """Initialize connection to Ollama, auto-selecting the best available vision model."""
        import requests
        import subprocess
        import time
        import sys

        def is_ollama_running():
            try:
                r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
                r.raise_for_status()
                return True
            except Exception:
                return False

        if not is_ollama_running():
            logger.info("Ollama server not detected – attempting to start it...")
            try:
                if sys.platform == "win32":
                    subprocess.Popen(
                        ["ollama", "serve"],
                        creationflags=subprocess.CREATE_NO_WINDOW,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    subprocess.Popen(
                        ["ollama", "serve"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                for _ in range(15):
                    time.sleep(1)
                    if is_ollama_running():
                        break
                else:
                    raise RuntimeError("Ollama server started but never became responsive")
            except Exception as e:
                raise RuntimeError(f"Failed to start Ollama server: {e}")

        # Discover installed models
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            installed = {m["name"] for m in r.json().get("models", [])}
        except Exception as e:
            raise ConnectionError(f"Cannot list Ollama models: {e}")

        # Vision model preference order — best for floor plan analysis first
        VISION_CANDIDATES = [
            "gemma3:4b",
            "qwen3-vl:latest", "qwen3-vl:8b", "qwen3-vl:4b", "qwen3-vl:2b",
            "qwen2.5vl:7b", "qwen2.5vl:3b",
            "llava:13b", "llava:7b",
            "llava-llama3:8b", "llava-phi3:3.8b",
            "llama3.2-vision:11b",
            "minicpm-v:8b",
            "moondream:latest", "moondream",
            "bakllava:7b",
        ]

        # Honour OLLAMA_MODEL env var if set and installed
        env_model = os.getenv("OLLAMA_MODEL", "").strip()
        if env_model:
            VISION_CANDIDATES.insert(0, env_model)

        chosen = next((m for m in VISION_CANDIDATES if m in installed), None)
        if chosen is None:
            raise ConnectionError(
                f"No supported vision model found in Ollama. "
                f"Installed: {sorted(installed)}. "
                f"Run: ollama pull qwen3-vl"
            )

        # Test the chosen model with a generous timeout (first call loads weights)
        test_payload = {
            "model": chosen,
            "prompt": "hi",
            "stream": False,
            "options": {"num_predict": 1},
        }
        try:
            resp = requests.post(f"{OLLAMA_URL}/api/generate", json=test_payload, timeout=60)
            resp.raise_for_status()
        except Exception as e:
            raise ConnectionError(f"Ollama model {chosen!r} failed test call: {e}")

        self.model_id = chosen
        self.client   = requests.Session()
        logger.info(f"✓ Ollama ready – model: {self.model_id}")

    # ------------------------------------------------------------------
    # Gemini rate limiter
    # ------------------------------------------------------------------
    @staticmethod
    def _gemini_wait():
        """Block until enough time has passed since the last Gemini API call."""
        global _GEMINI_LAST_CALL_TS
        with _GEMINI_LOCK:
            now = time.monotonic()
            elapsed = now - _GEMINI_LAST_CALL_TS
            if elapsed < _GEMINI_MIN_INTERVAL:
                wait = _GEMINI_MIN_INTERVAL - elapsed
                logger.info(f"Gemini rate limiter: waiting {wait:.1f}s (free tier 5 rpm)")
                time.sleep(wait)
            _GEMINI_LAST_CALL_TS = time.monotonic()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    async def analyze(
        self,
        image_data: Dict,
        detected_elements: Dict,
        grid_info: Dict,
    ) -> Dict:
        """
        Analyze floor plan with the configured AI backend.

        Args:
            image_data        : Processed image (numpy array under key 'image')
            detected_elements : Elements from YOLO detector (pixel coordinates)
            grid_info         : Grid info dict from GridDetector.detect()

        Returns:
            Enriched and validated element dictionary
        """
        if self.backend is None:
            logger.warning("Semantic analysis skipped — no AI backend configured. Returning YOLO detections unchanged.")
            return detected_elements

        logger.info(f"Semantic analysis via {self.backend}...")

        pil_image = Image.fromarray(image_data["image"])

        if self.backend == "ollama":
            prompt = self._create_prompt_ollama_simple(detected_elements, grid_info)
            response_text = self._call_ollama_text(prompt)
        elif self.backend == "gemini_api":
            prompt = self._create_prompt(detected_elements, grid_info)
            response_text = self._call_gemini(prompt, pil_image)
        elif self.backend == "anthropic_api":
            prompt = self._create_prompt(detected_elements, grid_info)
            response_text = self._call_anthropic(prompt, pil_image)
        else:
            # Local model stubs – should not reach here
            raise NotImplementedError(f"Backend '{self.backend}' not yet activated.")

        return await self._parse_and_merge(response_text, detected_elements)

    def read_element_annotation(self, pil_crop: Image.Image) -> dict:
        """
        Ask the vision LLM to read a structural annotation from a small image crop.
        Used as Pass 3 fallback when the PDF text layer has no annotation text.
        """
        prompt = (
            "You are reading a structural engineering drawing. "
            "This cropped image shows ONE column element and its annotation.\n"
            "The annotation text may be drawn as vector strokes — read it carefully.\n\n"
            "Return ONLY the following JSON, no markdown, no explanation:\n"
            '{"type_mark": "<label like C1, K5, or null>", '
            '"width_mm": <number or null>, '
            '"depth_mm": <number or null>, '
            '"is_circular": <true or false>, '
            '"diameter_mm": <number or null>}\n\n'
            "Rules:\n"
            "- '800x800' or '800×800' means width_mm=800, depth_mm=800, is_circular=false\n"
            "- '300∅' or 'Ø300' or '300dia' means is_circular=true, diameter_mm=300\n"
            "- If you cannot clearly read any dimensions, return numeric fields as null.\n"
            "- type_mark is the alphanumeric label such as C1, C2, B3.\n"
            "- Do not guess. Only return values you can clearly see."
        )
        try:
            text = self._call_vision_simple(prompt, pil_crop)
            data = self._parse_json_response(text)
            
            # Sanitize numeric fields to handle string responses (e.g., "800x800")
            for key in ["width_mm", "depth_mm", "diameter_mm"]:
                val = data.get(key)
                if isinstance(val, str):
                    # Special handling for "800x800" in width_mm
                    if key == "width_mm" and any(x in val for x in ["x", "X", "×"]):
                        clean_val = val.replace("×", "x").replace("X", "x")
                        parts = clean_val.split("x")
                        if len(parts) >= 2:
                            try:
                                data["width_mm"] = float(parts[0].strip())
                                data["depth_mm"] = float(parts[1].strip())
                                continue
                            except ValueError:
                                pass
                    
                    # Attempt to extract a single float
                    try:
                        clean_num = "".join(c for c in val if c.isdigit() or c == ".")
                        data[key] = float(clean_num) if clean_num else None
                    except ValueError:
                        data[key] = None
            
            return data
        except Exception as e:
            logger.debug(f"read_element_annotation failed: {e}")
            return {}

    def _call_vision_simple(self, prompt: str, image: Image.Image) -> str:
        """
        Lightweight single-turn vision call for simple structured queries.
        Uses low token budget (256) vs the full semantic analysis (32 768).
        """
        if self.backend == "gemini_api":
            self._gemini_wait()
            config_kwargs = dict(
                temperature=0.0,
                max_output_tokens=256,
                response_mime_type="application/json",
            )
            try:
                config_kwargs["thinking_config"] = self._genai_types.ThinkingConfig(
                    thinking_budget=0
                )
            except AttributeError:
                pass
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[prompt, image],
                config=self._genai_types.GenerateContentConfig(**config_kwargs),
            )
            return response.text

        elif self.backend == "anthropic_api":
            import base64
            import io
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=85)
            b64 = base64.standard_b64encode(buf.getvalue()).decode()
            msg = self.client.messages.create(
                model=self.model_id,
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64", "media_type": "image/jpeg", "data": b64,
                        }},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return msg.content[0].text

        elif self.backend == "ollama":
            return self._call_ollama(prompt, image, max_tokens=256)

        return "{}"

    def _parse_json_response(self, text: str) -> dict:
        """Strip markdown fences and parse JSON from an LLM response."""
        raw = text.strip()
        if "```" in raw:
            import re as _re
            raw = _re.sub(r"```(?:json)?\n?", "", raw).strip()
        return json.loads(raw)

    # ------------------------------------------------------------------
    # Backend call implementations
    # ------------------------------------------------------------------
    def _call_gemini(self, prompt: str, image: Image.Image, max_tokens: int = 32768) -> str:
        """Call Google Gemini API (rate-limited)."""
        self._gemini_wait()
        config_kwargs = dict(
            temperature=0.1,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        )
        try:
            config_kwargs["thinking_config"] = self._genai_types.ThinkingConfig(
                thinking_budget=0
            )
        except AttributeError:
            pass

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[prompt, image],
            config=self._genai_types.GenerateContentConfig(**config_kwargs),
        )
        try:
            candidate = response.candidates[0] if response.candidates else None
            if candidate and hasattr(candidate, "finish_reason"):
                logger.debug(f"Gemini finish_reason: {candidate.finish_reason}")
        except Exception:
            pass
        return response.text

    def _call_anthropic(self, prompt: str, image: Image.Image, max_tokens: int = 4000) -> str:
        """Call Anthropic Claude API with vision."""
        import base64
        import io

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        b64_image = base64.standard_b64encode(buf.getvalue()).decode()

        message = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64_image,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return message.content[0].text

    def _stream_ollama(self, payload: dict) -> str:
        """POST payload to Ollama /api/generate and return the full streamed text.

        Streaming keeps the connection alive token-by-token, so the per-chunk
        read timeout (120 s) is far more forgiving than a single timeout for the
        entire response — critical for large models running partly on CPU.
        """
        resp = self.client.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=(10, 120),
            stream=True,
        )
        resp.raise_for_status()
        parts = []
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                parts.append(chunk.get("response", ""))
                if chunk.get("done"):
                    break
        return "".join(parts)

    def _call_ollama(self, prompt: str, image: Image.Image, max_tokens: int = 32768) -> str:
        """Call Ollama with a vision prompt and image."""
        import base64
        import io

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        b64_image = base64.standard_b64encode(buf.getvalue()).decode()

        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "images": [b64_image],
            "stream": True,
            "options": {"temperature": 0.1, "num_predict": max_tokens},
        }
        try:
            return self._stream_ollama(payload)
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------
    def _create_prompt(self, detected_elements: Dict, grid_info: Dict) -> str:
        n_vlines = len(grid_info.get("x_lines_px", []))
        n_hlines = len(grid_info.get("y_lines_px", []))
        grid_src = grid_info.get("source", "unknown")
        x_sp = grid_info.get("x_spacings_mm", [])
        y_sp = grid_info.get("y_spacings_mm", [])
        span_x = f"{sum(x_sp):.0f} mm" if x_sp else "unknown"
        span_y = f"{sum(y_sp):.0f} mm" if y_sp else "unknown"

        return f"""You are an expert architectural analyst. Analyze this floor plan image and validate/enrich the detected elements.

Detected Data:
- Structural grid: {n_vlines} vertical × {n_hlines} horizontal lines (source: {grid_src})
- Grid extents: {span_x} wide × {span_y} deep (real-world, from grid dimension annotations)
- Walls: {len(detected_elements.get('walls', []))} detected
- Doors: {len(detected_elements.get('doors', []))} detected
- Windows: {len(detected_elements.get('windows', []))} detected
- Rooms: {len(detected_elements.get('rooms', []))} detected
- Columns: {len(detected_elements.get('columns', []))} detected

Task: Provide architectural analysis in strict JSON format.
Only include properties that can be clearly inferred from the floor plan image.
Do NOT guess or hallucinate fire ratings, compliance status, or accessibility details.

Required JSON schema:
{{
  "building_type": "residential/commercial/industrial/mixed",
  "construction_type": "concrete/timber/steel/masonry",
  "floor_count": 1,
  "validated_elements": {{
    "walls": [
      {{
        "id": 0,
        "type": "exterior/interior",
        "material": "concrete/brick/gypsum/masonry",
        "structural": true
      }}
    ],
    "doors": [
      {{
        "id": 0,
        "purpose": "main_entrance/bedroom/bathroom/corridor/service",
        "swing_direction": "Left/Right"
      }}
    ],
    "windows": [
      {{
        "id": 0,
        "orientation": "north/south/east/west/unknown",
        "operable": true
      }}
    ],
    "rooms": [
      {{
        "id": 0,
        "name": "Living Room",
        "purpose": "living/bedroom/kitchen/bathroom/corridor/office/storage",
        "area_sqm": 25.5,
        "ceiling_height": 2800
      }}
    ],
    "columns": [
      {{
        "id": 0,
        "type_mark": "C1",
        "shape": "rectangular/circular",
        "width_mm": 800,
        "depth_mm": 800,
        "diameter_mm": null,
        "material": "concrete/steel",
        "structural": true
      }}
    ]
  }},
  "inferred_properties": {{
    "total_floor_area": 120.5,
    "suggested_improvements": []
  }},
  "quality_checks": {{
    "walls_form_closed_rooms": true,
    "doors_properly_placed": true,
    "structural_integrity": "good/check_required",
    "issues_found": []
  }}
}}

Column annotation reading (CRITICAL):
  Floor plans print a type mark and dimensions beside each column symbol.
  Read these text labels carefully:
  - Rectangular: "C1 800x800", "C1 800×800", "C2 400x600"  → width_mm / depth_mm
  - Circular:    "C20 Ø200", "C20 ⌀300", "C20 200 dia"     → diameter_mm (set shape to "circular")
  - Type mark only: "C1" with separate dimension text nearby → still extract both
  If a column has no readable annotation, omit width_mm / depth_mm / diameter_mm.

Dimension guidelines (use as reference when estimating unannotated columns):
  - Square columns use standard sizes: {", ".join(str(s) for s in STANDARD_SQUARE_COLUMN_SIZES)} mm
  - Rectangular columns have varied dimensions: 250×600, 300×750, 400×900, 450×1000, etc.
  - Circular columns use standard diameters: {", ".join(str(s) for s in STANDARD_CIRCULAR_COLUMN_DIAMETERS)} mm
  Report exact values when readable from the drawing; otherwise estimate and prefer the nearest standard size.

IMPORTANT: Respond with ONLY valid JSON. No preamble, no explanation, just the JSON object."""

    def _create_prompt_text_only(self, detected_elements: Dict, grid_info: Dict) -> str:
        """
        Minimal prompt for text-only models (no image, e.g. qwen2.5-instruct).

        Gives the model the already-parsed structural data and asks it only to:
          • Pick one building_type and construction_type
          • Return the column list verbatim (width_mm / depth_mm already filled by PDF parser)
        Schema is intentionally tiny to prevent hallucination of placeholder strings.
        """
        cols = detected_elements.get("columns", [])
        col_entries = []
        for i, c in enumerate(cols):
            w     = c.get("width_mm")    or c.get("w_mm")    or "null"
            d     = c.get("depth_mm")    or c.get("d_mm")    or "null"
            diam  = c.get("diameter_mm") or "null"
            mark  = c.get("type_mark")   or c.get("mark")    or f"C{i+1}"
            shape = "circular" if c.get("diameter_mm") else "rectangular"
            col_entries.append(
                f'{{"id":{i},"type_mark":"{mark}","shape":"{shape}",'
                f'"width_mm":{w},"depth_mm":{d},"diameter_mm":{diam}}}'
            )
        cols_json = "[" + ",".join(col_entries) + "]"

        x_sp  = grid_info.get("x_spacings_mm", [])
        y_sp  = grid_info.get("y_spacings_mm", [])
        span_x = f"{sum(x_sp):.0f}" if x_sp else "unknown"
        span_y = f"{sum(y_sp):.0f}" if y_sp else "unknown"

        # Build the full expected JSON as a concrete template so the model
        # just replaces the two placeholder fields — minimises hallucination
        # and avoids the mis-structured output (bare list, unclosed dict, etc.)
        template = (
            '{\n'
            '  "building_type": "commercial",\n'
            '  "construction_type": "concrete",\n'
            '  "floor_count": 1,\n'
            f'  "validated_elements": {{"columns": {cols_json}}},\n'
            '  "inferred_properties": {"total_floor_area": null}\n'
            '}'
        )
        return (
            "You are a structural engineering assistant. "
            "Output ONLY the JSON object below — replace building_type and "
            "construction_type with the correct values, keep everything else identical.\n\n"
            f"Context: grid {span_x}mm × {span_y}mm, "
            f"{len(cols)} structural columns.\n\n"
            f"{template}\n\n"
            "building_type must be one of: residential / commercial / industrial / mixed\n"
            "construction_type must be one of: concrete / steel / timber / masonry\n"
            "Do NOT alter any numeric values. Output ONLY the JSON, nothing else."
        )

    def _create_prompt_ollama_simple(self, detected_elements: Dict, grid_info: Dict) -> str:
        """
        Ultra-minimal prompt for Ollama vision models used as text-only.

        These models (moondream, llava) have small context windows and cannot
        reliably follow a large template.  Ask ONLY for the two classification
        fields; the column data is already populated from the PDF parser and
        does not need to pass through the LLM.
        """
        x_sp  = grid_info.get("x_spacings_mm", [])
        y_sp  = grid_info.get("y_spacings_mm", [])
        span_x = f"{sum(x_sp):.0f}" if x_sp else "unknown"
        span_y = f"{sum(y_sp):.0f}" if y_sp else "unknown"
        n_cols = len(detected_elements.get("columns", []))

        return (
            f"Building: grid {span_x}mm × {span_y}mm, {n_cols} structural columns.\n"
            "Output ONLY this JSON, nothing else:\n"
            '{"building_type": "commercial", "construction_type": "concrete"}\n\n'
            "Replace the two values:\n"
            "building_type → one of: residential / commercial / industrial / mixed\n"
            "construction_type → one of: concrete / steel / timber / masonry"
        )

    # ------------------------------------------------------------------
    # Revit warning feedback loop
    # ------------------------------------------------------------------
    async def analyze_revit_warnings(
        self, warnings: List[str], recipe: Dict
    ) -> Dict:
        """
        Ask the AI to analyze Revit build warnings and suggest targeted
        corrections to the transaction JSON (recipe).

        Returns a dict with "corrections" and "summary", or {} on failure.
        """
        if not warnings:
            return {}

        prompt = self._create_revit_warning_prompt(warnings, recipe)
        try:
            if self.backend == "gemini_api":
                response_text = self._call_gemini_text(prompt)
            elif self.backend == "anthropic_api":
                response_text = self._call_anthropic_text(prompt)
            elif self.backend == "ollama":
                response_text = self._call_ollama_text(prompt)
            else:
                logger.warning("analyze_revit_warnings: local backends not supported")
                return {}
        except Exception as e:
            logger.warning(f"analyze_revit_warnings AI call failed: {e}")
            return {}

        return self._parse_corrections(response_text)

    def _create_revit_warning_prompt(self, warnings: List[str], recipe: Dict) -> str:
        """Build the prompt that asks the AI to fix Revit warnings."""
        warnings_block = "\n".join(f"  {i+1}. {w}" for i, w in enumerate(warnings))

        cols = recipe.get("columns", [])
        walls = recipe.get("walls", [])
        col_summary = json.dumps(
            [{"index": i, "width": c.get("width"), "depth": c.get("depth"),
              "height": c.get("height")} for i, c in enumerate(cols)],
            indent=2
        )
        wall_summary = json.dumps(
            [{"index": i, "thickness": w.get("thickness"),
              "height": w.get("height")} for i, w in enumerate(walls)],
            indent=2
        )

        return f"""You are a structural BIM engineer. Revit 2023 issued the following build warnings:

{warnings_block}

Current element dimensions (from the transaction JSON):
Columns ({len(cols)} total):
{col_summary}

Walls ({len(walls)} total):
{wall_summary}

Your task: suggest the minimum changes to fix the warnings above.
Only change fields that are clearly causing the warning.
Do NOT change anything unrelated to the warnings.

Respond with ONLY valid JSON matching this schema — no preamble, no markdown:
{{
  "corrections": [
    {{
      "element_type": "columns",
      "element_index": 0,
      "field": "width",
      "new_value": 400,
      "reason": "short explanation"
    }}
  ],
  "summary": "one-sentence description of all changes"
}}

If you cannot determine a safe correction, return:
{{"corrections": [], "summary": "no actionable corrections found"}}"""

    def _call_gemini_text(self, prompt: str) -> str:
        """Call Gemini with a text-only prompt."""
        self._gemini_wait()
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[prompt],
            config=self._genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=2048,
                response_mime_type="application/json",
            ),
        )
        return response.text

    def _call_anthropic_text(self, prompt: str) -> str:
        """Call Anthropic Claude with a text-only prompt."""
        message = self.client.messages.create(
            model=self.model_id,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def _call_ollama_text(self, prompt: str, max_tokens: int = 2048) -> str:
        """Call Ollama with a text-only prompt (no image)."""
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.0, "num_predict": max_tokens},
        }
        try:
            return self._stream_ollama(payload)
        except Exception as e:
            logger.error(f"Ollama text call failed: {e}")
            raise

    def _parse_corrections(self, response_text: str) -> Dict:
        """Parse the AI corrections JSON; return {} on failure."""
        try:
            cleaned = response_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)
            if "corrections" in data:
                return data
        except Exception as e:
            logger.warning(f"Could not parse Revit corrections JSON: {e}")
        return {}

    # ------------------------------------------------------------------
    # Response parsing & merging
    # ------------------------------------------------------------------
    def _repair_json(self, text: str) -> str:
        """
        Best-effort repair of a truncated or slightly malformed JSON string.
        (Implementation unchanged from original.)
        """
        import re as _re

        # Pre-clean
        text = _re.sub(r'\bundefined\b', 'null', text)
        text = _re.sub(r'\bNaN\b', 'null', text)
        text = _re.sub(r'\bInfinity\b', 'null', text)
        text = _re.sub(r',\s*([}\]])', r'\1', text)   # trailing commas

        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        stack: List[str] = []
        in_string = False
        escape = False
        depth_close_pos: dict = {}

        for i, ch in enumerate(text):
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in '{[':
                stack.append(ch)
            elif ch in '}]':
                # Only pop when the bracket type matches; otherwise the closer
                # is misplaced (e.g. a bare `]` closing an unclosed `{` dict)
                # and we skip it so the stack tracking stays accurate.
                if stack and ((ch == '}' and stack[-1] == '{') or
                              (ch == ']' and stack[-1] == '[')):
                    stack.pop()
                depth_close_pos[len(stack)] = i + 1

        if not stack:
            return text

        def _close(s):
            return ''.join('}' if b == '{' else ']' for b in reversed(s))

        # Truncated mid-string
        if in_string:
            candidate = text + '"' + _close(stack)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        # Roll back to last complete element at parent depth
        rollback_depth = len(stack) - 1
        if rollback_depth in depth_close_pos:
            pos = depth_close_pos[rollback_depth]
            truncated = _re.sub(r',\s*$', '', text[:pos].rstrip())
            candidate = truncated + _close(stack[:-1])
            try:
                json.loads(candidate)
                logger.debug(
                    f"JSON repair: rolled back to depth {rollback_depth}, "
                    f"recovered {pos} chars"
                )
                return candidate
            except json.JSONDecodeError:
                pass

        # Last resort
        return text + _close(stack)

    @staticmethod
    def _sanitize_json_text(text: str) -> str:
        """
        Pre-process raw LLM output so it can be parsed by json.loads.

        Handles common quirks from text-only local models (e.g. qwen2.5):
          • Strips markdown code fences  (```json … ```)
          • Strips Python/shell inline comments  (... # comment)
          • Strips C++ inline comments          (... // comment)
          • Removes trailing commas before ] or }
          • Inserts missing commas between adjacent JSON values/keys
          • Unwraps single-element array wrapping [ {...} ] → {...}
        """
        import re
        s = text.replace("```json", "").replace("```", "").strip()
        # Strip inline comments — must not touch URLs ("http://...")
        s = re.sub(r'\s*#(?![^"]*"[^"]*$)[^\n]*', '', s)   # Python # comments
        s = re.sub(r'\s*//(?![^"]*"[^"]*$)[^\n]*', '', s)  # C++ // comments
        # Remove trailing commas before closing bracket / brace
        s = re.sub(r',\s*([}\]])', r'\1', s)
        # Insert missing commas between adjacent JSON values/keys.
        # Matches: closing " or digit or } or ] followed by newline + optional
        # whitespace + another value opener (" { [) — the missing comma case.
        s = re.sub(r'(["\d}\]])([ \t]*\n[ \t]*)(?=["{[\d])', r'\1,\2', s)
        # Also handle bare keywords (null / true / false) missing a comma
        s = re.sub(r'(null|true|false)([ \t]*\n[ \t]*)(?=")', r'\1,\2', s)
        # Unwrap single-element array: [{...}] → {...}
        m = re.match(r'^\[\s*(\{.*\})\s*\]$', s, re.DOTALL)
        if m:
            s = m.group(1)
        return s.strip()

    async def _parse_and_merge(self, response_text: str, detected_elements: Dict) -> Dict:
        """Parse JSON response and merge with detected elements."""
        try:
            cleaned = self._sanitize_json_text(response_text)
            analysis = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.debug(f"AI JSON minor syntax issue ({e}) — attempting auto-repair")
            try:
                repaired = self._repair_json(cleaned)
                analysis = json.loads(repaired)
                logger.debug("JSON auto-repair succeeded")
            except Exception as e2:
                logger.warning(f"AI JSON repair failed ({e2}) — using detected elements as-is")
                return detected_elements

        if not isinstance(analysis, dict):
            # Handle case where LLM returns a list instead of a dict
            if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                logger.debug("AI response was a single-element array — unwrapped to dict.")
                analysis = analysis[0]
            else:
                logger.error(f"AI returned invalid JSON structure (expected dict, got {type(analysis).__name__})")
                return detected_elements

        enriched = detected_elements.copy()

        enriched["metadata"] = {
            "building_type": analysis.get("building_type"),
            "construction_type": analysis.get("construction_type"),
            "total_floor_area": analysis.get("inferred_properties", {}).get("total_floor_area"),
            "semantic_backend": self.backend,
        }

        validated_elements = analysis.get("validated_elements", {})
        # Local models (e.g. qwen2.5 text-only) sometimes return
        # validated_elements as a list of category dicts instead of a dict.
        # Flatten it into a single dict so the .get(key) calls below work.
        if isinstance(validated_elements, list):
            merged: Dict = {}
            for item in validated_elements:
                if isinstance(item, dict):
                    merged.update(item)
            validated_elements = merged
        if not isinstance(validated_elements, dict):
            validated_elements = {}

        for key in ("walls", "doors", "windows", "rooms", "columns"):
            validated = validated_elements.get(key, [])
            if not isinstance(validated, list):
                validated = []
            for i, element in enumerate(enriched.get(key, [])):
                if i < len(validated) and isinstance(validated[i], dict):
                    element.update(validated[i])

        logger.info(f"✓ Semantic analysis complete ({self.backend})")
        return enriched