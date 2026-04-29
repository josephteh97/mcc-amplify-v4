"""
YOLO Tiling Inference — extracted from PipelineOrchestrator.

Tiling strategy mirrors inspect_detections.ipynb:
  1. Apply CLAHE contrast enhancement
  2. Upsample to 300 DPI equivalent if rendered below target
  3. Slice into 1280×1280 px tiles with 200 px overlap
  4. Run YOLO at imgsz=1280 (scale=1.0×, no internal rescaling)
  5. Map detections back to global pixel coordinates
  6. Merge with NMS
  7. Filter by squareness + size

Coordinate contract: outputs are in original (pre-upsample) pixel space.
"""

import numpy as np
from loguru import logger
from pathlib import Path


def _enhance_for_yolo(img_rgb: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE contrast enhancement so that faint engineering-drawing
    lines become clearly visible before YOLO inference.
    """
    try:
        import cv2
        lab   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l     = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    except Exception:
        p1, p99 = np.percentile(img_rgb, [1, 99])
        if p99 > p1:
            enhanced = np.clip(
                (img_rgb.astype(float) - p1) / (p99 - p1) * 255, 0, 255
            ).astype(np.uint8)
        else:
            enhanced = img_rgb
    return enhanced


def load_yolo(weights_path: str | Path | None = None):
    """Load and return a YOLO model, or None if weights are missing."""
    if weights_path is None:
        weights_path = Path(__file__).parent.parent / "ml" / "weights" / "column-detect.pt"
    else:
        weights_path = Path(weights_path)

    if not weights_path.exists():
        logger.warning(f"YOLO weights not found at {weights_path} — detection will be skipped")
        return None
    try:
        from ultralytics import YOLO
        model = YOLO(str(weights_path))
        logger.info(f"YOLO model loaded: {weights_path.name}")
        return model
    except Exception as e:
        logger.warning(f"YOLO load failed ({e}) — detection will be skipped")
        return None


def run_yolo(
    yolo_model,
    image_data: dict,
    element_type: str = "column",
    tile_size: int = 1280,
    overlap: int = 200,
    conf: float = 0.25,
    iou: float = 0.45,
    target_dpi: int = 300,
    min_squareness: float = 0.75,
    min_side: int = 10,
    max_side: int = 80,
    imgsz: int | None = None,
) -> list:
    """
    Tiling YOLO inference on a rendered PDF image.

    Parameters
    ----------
    yolo_model      : loaded YOLO model (or None → returns [])
    image_data      : dict with "image" (H×W×3 uint8), "width", "height", "dpi"
    element_type    : element type label written into each detection dict
    tile_size       : tile side in pixels (sliding-window crop size)
    overlap         : overlap between tiles in pixels
    conf            : confidence threshold
    iou             : NMS IoU threshold
    target_dpi      : training DPI — image is upsampled to this before inference
    min_squareness  : min(w,h)/max(w,h) filter; set 0.0 to disable (e.g. for beams)
    min_side        : minimum bbox side in pixels after scaling
    max_side        : maximum bbox side in pixels after scaling
    imgsz           : YOLO network input resolution. Must match the model's
                      training imgsz (column=1280, framing=640). Defaults to
                      tile_size (the column model's training imgsz). Mismatch
                      tanks recall — beams trained at 640 but inferred at 1280
                      appear ~2× larger than the network learned to recognise.

    Returns
    -------
    list of detection dicts in original pixel space.
    """
    if imgsz is None:
        imgsz = tile_size
    if yolo_model is None or image_data is None:
        return []
    try:
        import torch
        from PIL import Image
        from torchvision.ops import nms as torch_nms

        img_np     = image_data["image"]
        render_dpi = image_data.get("dpi", 150)

        # Enhance contrast
        enhanced = _enhance_for_yolo(img_np)

        # Upsample to target DPI equivalent if rendered below target
        if render_dpi < target_dpi * 0.85:
            scale  = target_dpi / render_dpi
            new_w  = int(enhanced.shape[1] * scale)
            new_h  = int(enhanced.shape[0] * scale)
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS
            pil_img = Image.fromarray(enhanced).resize((new_w, new_h), resample)
            logger.info(
                f"YOLO upsample: {enhanced.shape[1]}×{enhanced.shape[0]} "
                f"({render_dpi} DPI) → {new_w}×{new_h} ({target_dpi} DPI eq.)"
            )
            coord_scale = render_dpi / target_dpi
        else:
            pil_img     = Image.fromarray(enhanced)
            coord_scale = 1.0

        W, H = pil_img.size

        # Sliding-window tiling
        step      = tile_size - overlap
        raw_boxes, raw_confs = [], []
        ys = list(range(0, H, step))
        xs = list(range(0, W, step))
        total_tiles = len(xs) * len(ys)

        for y0 in ys:
            for x0 in xs:
                x1 = min(x0 + tile_size, W);  xa = max(0, x1 - tile_size)
                y1 = min(y0 + tile_size, H);  ya = max(0, y1 - tile_size)
                tile = pil_img.crop((xa, ya, x1, y1))

                res = yolo_model.predict(
                    source=tile, imgsz=imgsz,
                    conf=conf, iou=iou, verbose=False,
                )[0]

                for box, c in zip(res.boxes.xyxy.cpu().numpy(),
                                  res.boxes.conf.cpu().numpy()):
                    raw_boxes.append([
                        float(box[0]) + xa, float(box[1]) + ya,
                        float(box[2]) + xa, float(box[3]) + ya,
                    ])
                    raw_confs.append(float(c))

        logger.info(f"YOLO tiling: {total_tiles} tiles, {len(raw_boxes)} raw detections")

        if not raw_boxes:
            return []

        # Global NMS
        b_t  = torch.tensor(raw_boxes, dtype=torch.float32)
        c_t  = torch.tensor(raw_confs, dtype=torch.float32)
        keep = torch_nms(b_t, c_t, iou_threshold=iou).numpy()
        b_nms = b_t.numpy()[keep]
        c_nms = c_t.numpy()[keep]

        detections = []
        for box, conf_val in zip(b_nms, c_nms):
            x1, y1, x2, y2 = box
            w = x2 - x1;  h = y2 - y1
            sq = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            passes = (
                (min_squareness == 0.0 or sq >= min_squareness)
                and min_side <= w <= max_side
                and min_side <= h <= max_side
            )
            if passes:
                bx1 = float(x1 * coord_scale)
                by1 = float(y1 * coord_scale)
                bx2 = float(x2 * coord_scale)
                by2 = float(y2 * coord_scale)
                # Stable per-run ID so downstream warnings can refer to a
                # specific detection (e.g. "structural_framing_17").
                det_id = f"{element_type}_{len(detections)}"
                detections.append({
                    "id":         det_id,
                    "type":       element_type,
                    "bbox":       [bx1, by1, bx2, by2],
                    "center":     [(bx1 + bx2) / 2, (by1 + by2) / 2],
                    "confidence": float(conf_val),
                })

        logger.info(
            f"YOLO tiling: {len(raw_boxes)} raw → {len(keep)} after NMS "
            f"→ {len(detections)} {element_type}(s) after filter"
        )
        return detections

    except Exception as e:
        logger.warning(f"YOLO inference failed: {e} — continuing without detections")
        return []
