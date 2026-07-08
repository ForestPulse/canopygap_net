from pathlib import Path
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np


class S2S1GapFractionTileFolderDataset(Dataset):
    """
    Dataset that reads Sentinel-2 spline coeffs + Sentinel-1 + canopy gap fraction chips.

    Expected structure:
      root/
        spline/
          x0085_y0059_2018.tif   (C=220, H=256, W=256)
        S1/
          x0085_y0059_2018.tif   (C=2,   H=256, W=256)  [VH, VV]
        grfra/
          x0085_y0059_2018.tif   (C=1,   H=256, W=256)  gap fraction in percent
    """

    def __init__(
        self,
        root_dir,
        s2_subdir="spline",
        s1_subdir="S1",
        label_subdir="grfra",
        s2_divisor=10000.0,
        s2_clamp01=True,
        s1_nodata=-32768.0,
        s1_scale_factor=100.0,
        add_s1_ratio=True,
        label_divisor=100.0,
        check_shapes=True,
        transforms=None,
    ):
        self.root = Path(root_dir)
        self.s2_dir = self.root / s2_subdir
        self.s1_dir = self.root / s1_subdir
        self.label_dir = self.root / label_subdir

        if not self.s2_dir.exists():
            raise FileNotFoundError(f"Missing S2 directory: {self.s2_dir}")
        if not self.s1_dir.exists():
            raise FileNotFoundError(f"Missing S1 directory: {self.s1_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Missing label directory: {self.label_dir}")

        self.s2_divisor = float(s2_divisor)
        self.s2_clamp01 = bool(s2_clamp01)
        self.s1_nodata = s1_nodata
        self.s1_scale_factor = float(s1_scale_factor)
        self.add_s1_ratio = bool(add_s1_ratio)
        self.label_divisor = float(label_divisor)
        self.check_shapes = bool(check_shapes)
        self.transforms = transforms

        self.files = sorted([
            f for f in self.s2_dir.glob("*.tif")
            if (self.s1_dir / f.name).exists() and (self.label_dir / f.name).exists()
        ])

        if len(self.files) == 0:
            raise RuntimeError(f"No paired S2/S1/gap-fraction tiles found in {root_dir}")

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _read(path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            arr = src.read()
        return arr.astype(np.float32)

    def __getitem__(self, idx):
        s2_path = self.files[idx]
        s1_path = self.s1_dir / s2_path.name
        label_path = self.label_dir / s2_path.name

        # ---- S2 spline coefficients ----
        s2 = torch.from_numpy(self._read(s2_path)).float()  # expected (220,256,256)
        s2 = s2 / self.s2_divisor
        if self.s2_clamp01:
            s2 = torch.clamp(s2, 0.0, 1.0)

        # ---- S1 ----
        s1 = torch.from_numpy(self._read(s1_path)).float()  # expected (2,256,256)
        if s1.ndim == 2:
            s1 = s1.unsqueeze(0)

        if s1.shape[0] != 2:
            raise ValueError(
                f"Expected 2 S1 bands [VH,VV], got shape {tuple(s1.shape)} for {s1_path}"
            )

        if self.s1_nodata is not None:
            nodata_mask = s1 == float(self.s1_nodata)
        else:
            nodata_mask = torch.zeros_like(s1, dtype=torch.bool)

        s1 = s1 / self.s1_scale_factor
        s1 = torch.where(nodata_mask, torch.zeros_like(s1), s1)

        if self.add_s1_ratio:
            vh = s1[0:1]
            vv = s1[1:2]
            ratio = vh - vv
            s1 = torch.cat([s1, ratio], dim=0)  # (3,H,W)

        # ---- Gap fraction label ----
        label = torch.from_numpy(self._read(label_path)).float()
        if label.ndim == 2:
            label = label.unsqueeze(0)

        label = label / self.label_divisor
        label = torch.clamp(label, 0.0, 1.0)

        # ---- Shape checks ----
        if self.check_shapes:
            if s2.shape[1:] != label.shape[1:]:
                raise ValueError(
                    f"S2 and label shape mismatch for {s2_path.name}: "
                    f"s2={tuple(s2.shape)}, label={tuple(label.shape)}"
                )
            if s1.shape[1:] != s2.shape[1:]:
                raise ValueError(
                    f"S1 and S2 shape mismatch for {s2_path.name}: "
                    f"s1={tuple(s1.shape)}, s2={tuple(s2.shape)}"
                )

        sample = {
            "s2": s2,          # e.g. (220,256,256)
            "s1": s1,          # e.g. (3,256,256)
            "label": label,    # (1,256,256), fraction 0–1
            "name": s2_path.stem,
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample