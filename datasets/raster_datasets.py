from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class S2S1GapFractionNPZDataset(Dataset):

    def __init__(
        self,
        root_dir,
        s2_subdir="spline",
        s1_subdir="S1",
        label_subdir="canopygap",
        s2_divisor=10000.0,
        s2_clamp01=True,
        s1_nodata=-32768.0,
        s1_scale_factor=100.0,
        add_s1_ratio=True,
        label_divisor=100.0,
        transforms=None,
    ):
        self.root = Path(root_dir)
        self.s2_dir = self.root / s2_subdir
        self.s1_dir = self.root / s1_subdir
        self.label_dir = self.root / label_subdir

        self.s2_divisor = float(s2_divisor)
        self.s2_clamp01 = bool(s2_clamp01)
        self.s1_nodata = s1_nodata
        self.s1_scale_factor = float(s1_scale_factor)
        self.add_s1_ratio = bool(add_s1_ratio)
        self.label_divisor = float(label_divisor)
        self.transforms = transforms

        # Labels determine which samples are retained.
        self.samples = []

        for label_path in sorted(self.label_dir.rglob("*.npz")):
            relative_path = label_path.relative_to(self.label_dir)

            s2_path = self.s2_dir / relative_path
            s1_path = self.s1_dir / relative_path

            if not s2_path.is_file():
                raise FileNotFoundError(
                    f"Missing S2 chip for {relative_path}"
                )

            if not s1_path.is_file():
                raise FileNotFoundError(
                    f"Missing S1 chip for {relative_path}"
                )

            self.samples.append(
                (s2_path, s1_path, label_path)
            )

        if not self.samples:
            raise RuntimeError(
                f"No paired NPZ samples found under {self.root}"
            )

        print(f"Paired samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _read_npz(path, key):
        with np.load(path, allow_pickle=False) as archive:
            return np.asarray(
                archive[key],
                dtype=np.float32,
            )

    def __getitem__(self, idx):
        s2_path, s1_path, label_path = self.samples[idx]

        # S2 spline coefficients
        s2 = torch.from_numpy(
            self._read_npz(s2_path, "data")
        ).float()

        s2 = s2 / self.s2_divisor

        if self.s2_clamp01:
            s2 = torch.clamp(s2, 0.0, 1.0)

        # Sentinel-1
        s1 = torch.from_numpy(
            self._read_npz(s1_path, "data")
        ).float()

        if self.s1_nodata is not None:
            nodata_mask = s1 == float(self.s1_nodata)
        else:
            nodata_mask = torch.zeros_like(
                s1,
                dtype=torch.bool,
            )

        s1 = s1 / self.s1_scale_factor
        s1 = torch.where(
            nodata_mask,
            torch.zeros_like(s1),
            s1,
        )

        if self.add_s1_ratio:
            vh = s1[0:1]
            vv = s1[1:2]
            s1 = torch.cat(
                [s1, vh - vv],
                dim=0,
            )

        # Label
        label = torch.from_numpy(
            self._read_npz(label_path, "label")
        ).float()

        label = label / self.label_divisor
        label = torch.clamp(label, 0.0, 1.0)

        sample = {
            "s2": s2,
            "s1": s1,
            "label": label,
            "name": label_path.stem,
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample