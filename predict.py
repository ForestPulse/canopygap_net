import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import rasterio
from rasterio.windows import Window, from_bounds
from tqdm import tqdm

from models.gap_net import Sentinel2ResUNet
import config


PATCH = config.PREDICTION_PATCH_SIZE
STRIDE = PATCH // 2
BATCH = getattr(config, "PREDICTION_BATCH_SIZE", 4)
TILE = getattr(config, "PREDICTION_TILE_SIZE", 1024)
HALO = PATCH

OUT_NODATA = 255
INTERNAL_NODATA = -9999.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_hann2d(size: int, eps: float = 1e-6) -> torch.Tensor:
    w1d = torch.hann_window(size, periodic=False)
    w2d = torch.outer(w1d, w1d)
    w2d = w2d / (w2d.max() + eps)
    return torch.clamp(w2d, min=0.05).unsqueeze(0)  # (1,H,W)


blend_window = make_hann2d(PATCH)


def load_tile_list():
    tile_list_file = getattr(config, "PREDICTION_TILE_LIST_FILE", None)
    tile_list = getattr(config, "PREDICTION_TILE_LIST", None)

    if tile_list_file is not None:
        with open(tile_list_file, "r") as f:
            return set(line.strip() for line in f if line.strip())

    if tile_list is not None:
        return set(tile_list)

    return None


def pad_array_for_sliding(arr: np.ndarray, patch: int, stride: int, pad_value=0):
    c, h, w = arr.shape

    if h < patch:
        pad_h = patch - h
    else:
        rem_h = (h - patch) % stride
        pad_h = 0 if rem_h == 0 else (stride - rem_h)

    if w < patch:
        pad_w = patch - w
    else:
        rem_w = (w - patch) % stride
        pad_w = 0 if rem_w == 0 else (stride - rem_w)

    arr_pad = np.pad(
        arr,
        ((0, 0), (0, pad_h), (0, pad_w)),
        mode="constant",
        constant_values=pad_value,
    )
    return arr_pad, h, w


def build_valid_mask(image_np: np.ndarray, nodata_value, nodata_eps=0):
    """
    image_np: (C,H,W)
    returns: (H,W) bool where True means valid.
    Uses the S2 spline stack as reference validity mask.
    """
    if nodata_value is not None:
        if nodata_eps > 0:
            nodata = np.all(
                np.abs(image_np.astype(np.float32) - float(nodata_value)) <= nodata_eps,
                axis=0,
            )
        else:
            nodata = np.all(image_np == nodata_value, axis=0)
    else:
        # fallback: treat all-zero spline vectors as nodata
        nodata = np.all(image_np == 0, axis=0)

    return ~nodata


def derive_reference_grid(raster_paths):
    datasets = [rasterio.open(p) for p in raster_paths]
    first = datasets[0]

    resx = first.transform.a
    resy = -first.transform.e

    minx = min(ds.bounds.left for ds in datasets)
    maxx = max(ds.bounds.right for ds in datasets)
    miny = min(ds.bounds.bottom for ds in datasets)
    maxy = max(ds.bounds.top for ds in datasets)

    width = int(round((maxx - minx) / resx))
    height = int(round((maxy - miny) / resy))

    for ds in datasets:
        ds.close()

    return {
        "minx": minx,
        "maxy": maxy,
        "width": width,
        "height": height,
        "resx": resx,
        "resy": resy,
    }


def normalize_s2(tile_np: np.ndarray) -> torch.Tensor:
    """
    Match training: spline coefficients are divided by 10000 and clamped to [0,1].
    """
    t = torch.from_numpy(tile_np).float()
    t = torch.clamp(t / 10000.0, 0.0, 1.0)
    return t


def preprocess_s1_like_training(s1_patch_raw: torch.Tensor, s1_nodata, s1_scale_factor=100.0):
    """
    Match spline training loader:
      raw S1 -> /100 -> nodata to 0 -> add VH - VV channel.

    Input:  (2, PATCH, PATCH)
    Output: (3, PATCH, PATCH)
    """
    if s1_nodata is not None:
        nodata_mask = s1_patch_raw == float(s1_nodata)
    else:
        nodata_mask = torch.zeros_like(s1_patch_raw, dtype=torch.bool)

    s1 = s1_patch_raw / s1_scale_factor
    s1 = torch.where(nodata_mask, torch.zeros_like(s1), s1)

    if s1.shape[0] != 2:
        raise ValueError(f"Expected 2 S1 bands [VH,VV], got {tuple(s1.shape)}")

    vh = s1[0:1]
    vv = s1[1:2]
    ratio = vh - vv
    return torch.cat([s1, ratio], dim=0)


def load_model_for_inference(model_path: str):
    model = Sentinel2ResUNet(
        in_channels=config.NUM_BANDS,      # should be 220
        s1_in_channels=config.S1_BANDS,    # should be 3
    )

    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def predict_tile(model, s2_np, s1_np, s2_nodata, s1_nodata, nodata_eps=0):
    """
    s2_np: (220,H,W) spline coefficients
    s1_np: (2,H,W) S1 aligned to spline grid
    returns: (H,W) float32 gap fraction in [roughly 0,1], with INTERNAL_NODATA where invalid
    """
    valid_mask_np = build_valid_mask(s2_np, s2_nodata, nodata_eps=nodata_eps)

    if not valid_mask_np.any():
        _, h, w = s2_np.shape
        return np.ones((h, w), dtype=np.float32) * INTERNAL_NODATA

    s2 = normalize_s2(s2_np)
    s1_raw = torch.from_numpy(s1_np).float()

    _, h, w = s2.shape
    out_sum = torch.zeros((h, w), dtype=torch.float32)
    w_sum = torch.zeros((h, w), dtype=torch.float32)
    valid = torch.from_numpy(valid_mask_np.astype(np.float32))

    coords = []
    patches_s2 = []
    patches_s1 = []

    use_amp = device.type == "cuda"

    for i in range(0, h - PATCH + 1, STRIDE):
        for j in range(0, w - PATCH + 1, STRIDE):
            vm = valid[i:i + PATCH, j:j + PATCH]
            if vm.sum().item() == 0:
                continue

            s2_patch = s2[:, i:i + PATCH, j:j + PATCH]
            s1_patch_raw = s1_raw[:, i:i + PATCH, j:j + PATCH]
            s1_patch = preprocess_s1_like_training(s1_patch_raw, s1_nodata)

            coords.append((i, j))
            patches_s2.append(s2_patch)
            patches_s1.append(s1_patch)

            if len(patches_s2) == BATCH:
                _run_batch(model, coords, patches_s2, patches_s1, valid, out_sum, w_sum, use_amp)
                coords.clear()
                patches_s2.clear()
                patches_s1.clear()

    if patches_s2:
        _run_batch(model, coords, patches_s2, patches_s1, valid, out_sum, w_sum, use_amp)

    out = torch.empty((h, w), dtype=torch.float32)
    m = w_sum > 0
    out[m] = out_sum[m] / w_sum[m]
    out[~m] = INTERNAL_NODATA

    return out.numpy().astype(np.float32)


def _run_batch(model, coords, patches_s2, patches_s1, valid, out_sum, w_sum, use_amp):
    batch_s2 = torch.stack(patches_s2, dim=0).to(device, non_blocking=True)
    batch_s1 = torch.stack(patches_s1, dim=0).to(device, non_blocking=True)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        pred = model(batch_s2, batch_s1)

    pred = pred.float().detach().cpu()
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)

    for k, (ii, jj) in enumerate(coords):
        vm2 = valid[ii:ii + PATCH, jj:jj + PATCH]
        w_patch = blend_window[0] * vm2
        out_sum[ii:ii + PATCH, jj:jj + PATCH] += pred[k, 0] * w_patch
        w_sum[ii:ii + PATCH, jj:jj + PATCH] += w_patch


def gap_float_to_uint8(pred_center: np.ndarray):
    """
    Convert model output fraction to uint8 percent.
    Valid pixels: 0..100
    Nodata pixels: 255
    """
    valid = pred_center != INTERNAL_NODATA
    out = np.full(pred_center.shape, OUT_NODATA, dtype=np.uint8)
    out[valid] = np.rint(np.clip(pred_center[valid], 0.0, 1.0) * 100.0).astype(np.uint8)
    return out


def find_single_tif(folder: Path):
    tif_files = sorted(folder.glob("*.tif"))
    if len(tif_files) == 0:
        raise FileNotFoundError(f"No .tif found in {folder}")
    if len(tif_files) > 1:
        raise RuntimeError(f"Multiple .tif files found in {folder}, please disambiguate: {tif_files}")
    return tif_files[0]


def build_temp_vrt(raster_paths, vrt_path: Path, reference_grid=None):
    import xml.etree.ElementTree as ET

    if len(raster_paths) == 0:
        raise ValueError(f"No rasters provided for VRT {vrt_path}")

    datasets = [rasterio.open(p) for p in raster_paths]

    first = datasets[0]
    count = first.count
    dtype = first.dtypes[0]
    crs_wkt = first.crs.to_wkt()
    resx = first.transform.a
    resy = -first.transform.e

    for ds, path in zip(datasets, raster_paths):
        if ds.count != count:
            raise ValueError(f"Band count mismatch in {path}")
        if ds.crs.to_wkt() != crs_wkt:
            raise ValueError(f"CRS mismatch in {path}")
        if abs(ds.transform.a - resx) > 1e-9 or abs((-ds.transform.e) - resy) > 1e-9:
            raise ValueError(f"Resolution mismatch in {path}")

    if reference_grid is None:
        minx = min(ds.bounds.left for ds in datasets)
        maxx = max(ds.bounds.right for ds in datasets)
        miny = min(ds.bounds.bottom for ds in datasets)
        maxy = max(ds.bounds.top for ds in datasets)
        width = int(round((maxx - minx) / resx))
        height = int(round((maxy - miny) / resy))
    else:
        minx = reference_grid["minx"]
        maxy = reference_grid["maxy"]
        width = reference_grid["width"]
        height = reference_grid["height"]
        resx = reference_grid["resx"]
        resy = reference_grid["resy"]

    vrt = ET.Element("VRTDataset", rasterXSize=str(width), rasterYSize=str(height))

    srs = ET.SubElement(vrt, "SRS")
    srs.text = crs_wkt

    geo = ET.SubElement(vrt, "GeoTransform")
    geo.text = f"{minx},{resx},0,{maxy},0,{-resy}"

    for b in range(1, count + 1):
        band = ET.SubElement(vrt, "VRTRasterBand", dataType=dtype, band=str(b))

        for ds, path in zip(datasets, raster_paths):
            xoff = int(round((ds.bounds.left - minx) / resx))
            yoff = int(round((maxy - ds.bounds.top) / resy))

            src = ET.SubElement(band, "SimpleSource")

            fname = ET.SubElement(src, "SourceFilename", relativeToVRT="0")
            fname.text = str(path)

            ET.SubElement(src, "SourceBand").text = str(b)
            ET.SubElement(src, "SrcRect", xOff="0", yOff="0", xSize=str(ds.width), ySize=str(ds.height))
            ET.SubElement(src, "DstRect", xOff=str(xoff), yOff=str(yoff), xSize=str(ds.width), ySize=str(ds.height))

    tree = ET.ElementTree(vrt)
    tree.write(vrt_path)

    for ds in datasets:
        ds.close()

    return vrt_path


def collect_year_tile_pairs(spline_root: Path, s1_root: Path, year: int, tile_allow=None):
    spline_paths = []
    s1_paths = []
    tile_names = []

    tile_dirs = sorted([
        p for p in spline_root.iterdir()
        if p.is_dir() and p.name.startswith("X")
    ])

    if tile_allow is not None:
        tile_dirs = [p for p in tile_dirs if p.name in tile_allow]

    print(f"Candidate tiles for {year}: {len(tile_dirs)}")

    for tile_dir in tile_dirs:
        tile_name = tile_dir.name

        s2_path = tile_dir / f"ThermSpline_coefs_{year}.tif"
        if not s2_path.exists():
            print(f"Skipping {tile_name} {year}: missing spline file")
            continue

        s1_tile_dir = s1_root / str(year) / tile_name
        if not s1_tile_dir.exists():
            print(f"Skipping {tile_name} {year}: missing S1 folder")
            continue

        try:
            s1_path = find_single_tif(s1_tile_dir)
        except Exception as e:
            print(f"Skipping {tile_name} {year}: {e}")
            continue

        spline_paths.append(s2_path)
        s1_paths.append(s1_path)
        tile_names.append(tile_name)

    print(f"Usable tiles for {year}: {len(tile_names)}")
    return spline_paths, s1_paths, tile_names


def predict_one_raster_pair(model, s2_path: Path, s1_path: Path, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(s2_path) as s2_src, rasterio.open(s1_path) as s1_src:
        assert s2_src.crs == s1_src.crs, f"CRS mismatch:\n{s2_path}\n{s1_path}"
        assert s2_src.transform == s1_src.transform, f"Transform mismatch:\n{s2_path}\n{s1_path}"
        assert s2_src.width == s1_src.width and s2_src.height == s1_src.height, f"Shape mismatch:\n{s2_path}\n{s1_path}"

        if s2_src.count != config.NUM_BANDS:
            raise ValueError(f"{s2_path} has {s2_src.count} bands, expected {config.NUM_BANDS}")

        if s1_src.count != 2:
            raise ValueError(f"{s1_path} has {s1_src.count} bands, expected 2")

        s2_nodata = s2_src.nodata
        s1_nodata = s1_src.nodata
        h_total, w_total = s2_src.height, s2_src.width

        out_profile = {
            "driver": "GTiff",
            "height": h_total,
            "width": w_total,
            "count": 1,
            "dtype": rasterio.uint8,
            "crs": s2_src.crs,
            "transform": s2_src.transform,
            "nodata": OUT_NODATA,
            "compress": "lzw",
        }

        with rasterio.open(output_path, "w", **out_profile) as dst:
            for top in tqdm(range(0, h_total, TILE), desc=f"Predicting {s2_path.name}"):
                tile_h = min(TILE, h_total - top)

                for left in range(0, w_total, TILE):
                    tile_w = min(TILE, w_total - left)

                    r0 = max(0, top - HALO)
                    c0 = max(0, left - HALO)
                    r1 = min(h_total, top + tile_h + HALO)
                    c1 = min(w_total, left + tile_w + HALO)

                    window = Window(c0, r0, c1 - c0, r1 - r0)

                    s2_np = s2_src.read(window=window)
                    s1_np = s1_src.read(window=window)

                    orig_h = s2_np.shape[1]
                    orig_w = s2_np.shape[2]

                    s2_np, _, _ = pad_array_for_sliding(s2_np, PATCH, STRIDE, pad_value=0)
                    s1_np, _, _ = pad_array_for_sliding(s1_np, PATCH, STRIDE, pad_value=0)

                    pred = predict_tile(
                        model,
                        s2_np,
                        s1_np,
                        s2_nodata=s2_nodata,
                        s1_nodata=s1_nodata,
                        nodata_eps=0,
                    )

                    pred = pred[:orig_h, :orig_w]

                    inner_top = top - r0
                    inner_left = left - c0
                    pred_center = pred[
                        inner_top: inner_top + tile_h,
                        inner_left: inner_left + tile_w,
                    ]

                    if pred_center.shape != (tile_h, tile_w):
                        raise ValueError(
                            f"pred_center shape mismatch for {s2_path.name}: "
                            f"got {pred_center.shape}, expected {(tile_h, tile_w)}"
                        )

                    pred_center_u8 = gap_float_to_uint8(pred_center)

                    dst.write(
                        pred_center_u8,
                        1,
                        window=Window(left, top, tile_w, tile_h),
                    )


def chip_prediction_to_force_tiles(pred_path: Path, spline_root: Path, output_root: Path, year: int, tile_allow=None):
    tile_dirs = sorted([
        p for p in spline_root.iterdir()
        if p.is_dir() and p.name.startswith("X")
    ])

    if tile_allow is not None:
        tile_dirs = [p for p in tile_dirs if p.name in tile_allow]

    output_template = getattr(config, "PREDICTION_OUTPUT_FILENAME_TEMPLATE", "gapfraction_{year}.tif")

    with rasterio.open(pred_path) as pred_src:
        for tile_dir in tile_dirs:
            tile_name = tile_dir.name
            ref_tile = tile_dir / f"ThermSpline_coefs_{year}.tif"

            if not ref_tile.exists():
                continue

            out_name = output_template.format(year=year, tile=tile_name)
            out_path = output_root / str(year) / tile_name / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with rasterio.open(ref_tile) as ref_src:
                if pred_src.crs != ref_src.crs:
                    raise ValueError(f"CRS mismatch for tile {tile_name}")

                window = from_bounds(*ref_src.bounds, transform=pred_src.transform)
                window = window.round_offsets().round_lengths()

                arr = pred_src.read(1, window=window)

                if arr.shape != (ref_src.height, ref_src.width):
                    raise ValueError(
                        f"Chip shape mismatch for {tile_name}: "
                        f"got {arr.shape}, expected {(ref_src.height, ref_src.width)}"
                    )

                profile = ref_src.profile.copy()
                profile.update(
                    count=1,
                    dtype=rasterio.uint8,
                    nodata=OUT_NODATA,
                    compress="lzw",
                )

                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(arr.astype(np.uint8), 1)

            print(f"Wrote {out_path}")


def main():
    spline_root = Path(config.PREDICTION_SPLINE_ROOT)
    s1_root = Path(config.PREDICTION_S1_ROOT)
    output_root = Path(config.PREDICTION_OUTPUT_ROOT)
    years = getattr(config, "PREDICTION_YEARS", [2018])

    model = load_model_for_inference(config.PREDICTION_MODEL)
    tile_allow = load_tile_list()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for year in years:
            print(f"Preparing inputs for year {year}")

            spline_paths, s1_paths, tile_names = collect_year_tile_pairs(
                spline_root=spline_root,
                s1_root=s1_root,
                year=year,
                tile_allow=tile_allow,
            )

            if len(spline_paths) == 0:
                print(f"No valid tiles found for year {year}")
                continue

            spline_vrt = tmpdir / f"spline_{year}.vrt"
            s1_vrt = tmpdir / f"s1_{year}.vrt"

            print(f"Building spline VRT for {year}")
            ref_grid = derive_reference_grid(spline_paths)
            build_temp_vrt(spline_paths, spline_vrt, reference_grid=ref_grid)

            print(f"Building S1 VRT for {year}")
            build_temp_vrt(s1_paths, s1_vrt, reference_grid=ref_grid)

            pred_out = Path(config.PREDICTION_OUTPUT)
            if len(years) > 1:
                pred_out = pred_out.with_name(f"{pred_out.stem}_{year}{pred_out.suffix}")

            print(f"Predicting seamless gap-fraction raster for {year}")
            predict_one_raster_pair(model, spline_vrt, s1_vrt, pred_out)

            print(f"Chipping prediction back to FORCE tiles for {year}")
            chip_prediction_to_force_tiles(
                pred_path=pred_out,
                spline_root=spline_root,
                output_root=output_root,
                year=year,
                tile_allow=tile_allow,
            )


if __name__ == "__main__":
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    main()
