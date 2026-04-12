"""AnnData <-> Parquet split I/O utilities.

Layout (prefix optional):
- {prefix}X_csr_*.parquet
- {prefix}obs.parquet
- {prefix}var.parquet
- {prefix}obsm/{key}.parquet
- {prefix}varm/{key}.parquet
- {prefix}obsp/{key}_csr_*.parquet (or .parquet if dense)
- {prefix}varp/{key}_csr_*.parquet (or .parquet if dense)
- {prefix}layers/{key}_csr_*.parquet (or .parquet if dense)
- {prefix}uns.json (best-effort)
"""

from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np
import pandas as pd
from scipy import sparse

import anndata as ad


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _fname(dir_path: str, prefix: str, name: str) -> str:
    if prefix:
        return os.path.join(dir_path, f"{prefix}{name}")
    return os.path.join(dir_path, name)


def _write_csr(X: sparse.csr_matrix, base_path: str) -> None:
    pd.DataFrame({"data": X.data}).to_parquet(base_path + "_csr_data.parquet")
    pd.DataFrame({"indices": X.indices}).to_parquet(base_path + "_csr_indices.parquet")
    pd.DataFrame({"indptr": X.indptr}).to_parquet(base_path + "_csr_indptr.parquet")
    pd.DataFrame({"shape0": [X.shape[0]], "shape1": [X.shape[1]]}).to_parquet(
        base_path + "_csr_shape.parquet"
    )


def _read_csr(base_path: str) -> sparse.csr_matrix:
    data = pd.read_parquet(base_path + "_csr_data.parquet")["data"].to_numpy()
    indices = pd.read_parquet(base_path + "_csr_indices.parquet")["indices"].to_numpy()
    indptr = pd.read_parquet(base_path + "_csr_indptr.parquet")["indptr"].to_numpy()
    shape_df = pd.read_parquet(base_path + "_csr_shape.parquet")
    shape = (int(shape_df["shape0"].iloc[0]), int(shape_df["shape1"].iloc[0]))
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


def _write_matrix(value, base_path: str) -> None:
    if sparse.issparse(value):
        _write_csr(value.tocsr(), base_path)
    else:
        pd.DataFrame(value).to_parquet(base_path + ".parquet")


def _read_matrix(base_path: str):
    csr_path = base_path + "_csr_data.parquet"
    if os.path.exists(csr_path):
        return _read_csr(base_path)
    return pd.read_parquet(base_path + ".parquet").to_numpy()


def to_parquet(adata: ad.AnnData, out_dir: str, prefix: str = "") -> None:
    """Save AnnData to a split parquet directory.

    prefix: optional string prepended to top-level file names (e.g. "group_").
    """
    _ensure_dir(out_dir)

    # X
    if sparse.issparse(adata.X):
        _write_csr(adata.X.tocsr(), _fname(out_dir, prefix, "X"))
    else:
        pd.DataFrame(adata.X).to_parquet(_fname(out_dir, prefix, "X") + ".parquet")

    # obs / var
    adata.obs.to_parquet(_fname(out_dir, prefix, "obs") + ".parquet")
    adata.var.to_parquet(_fname(out_dir, prefix, "var") + ".parquet")

    # obsm / varm
    if len(adata.obsm):
        obsm_dir = _fname(out_dir, prefix, "obsm")
        _ensure_dir(obsm_dir)
        for k, v in adata.obsm.items():
            pd.DataFrame(v).to_parquet(os.path.join(obsm_dir, f"{k}.parquet"))

    if len(adata.varm):
        varm_dir = _fname(out_dir, prefix, "varm")
        _ensure_dir(varm_dir)
        for k, v in adata.varm.items():
            pd.DataFrame(v).to_parquet(os.path.join(varm_dir, f"{k}.parquet"))

    # obsp / varp
    if len(adata.obsp):
        obsp_dir = _fname(out_dir, prefix, "obsp")
        _ensure_dir(obsp_dir)
        for k, v in adata.obsp.items():
            _write_matrix(v, os.path.join(obsp_dir, k))

    if len(adata.varp):
        varp_dir = _fname(out_dir, prefix, "varp")
        _ensure_dir(varp_dir)
        for k, v in adata.varp.items():
            _write_matrix(v, os.path.join(varp_dir, k))

    # layers
    if len(adata.layers):
        layers_dir = _fname(out_dir, prefix, "layers")
        _ensure_dir(layers_dir)
        for k, v in adata.layers.items():
            _write_matrix(v, os.path.join(layers_dir, k))

    # uns (best-effort)
    if len(adata.uns):
        uns_path = _fname(out_dir, prefix, "uns.json")
        with open(uns_path, "w", encoding="utf-8") as f:
            json.dump(adata.uns, f, ensure_ascii=True, default=str)


def from_parquet(in_dir: str, prefix: str = "", verbose: bool = True) -> ad.AnnData:
    """Load AnnData from a split parquet directory.

    prefix: optional string prepended to top-level file names (e.g. "group_").
    """
    detected: Dict[str, str] = {}

    # X
    x_base = _fname(in_dir, prefix, "X")
    if os.path.exists(x_base + "_csr_data.parquet"):
        detected["X"] = x_base + "_csr_*.parquet"
        X = _read_csr(x_base)
    else:
        detected["X"] = x_base + ".parquet"
        X = pd.read_parquet(x_base + ".parquet").to_numpy()

    obs_path = _fname(in_dir, prefix, "obs") + ".parquet"
    var_path = _fname(in_dir, prefix, "var") + ".parquet"
    detected["obs"] = obs_path
    detected["var"] = var_path
    obs = pd.read_parquet(obs_path)
    var = pd.read_parquet(var_path)

    adata = ad.AnnData(X=X, obs=obs, var=var)

    # obsm / varm
    obsm_dir = _fname(in_dir, prefix, "obsm")
    if os.path.isdir(obsm_dir):
        for fn in os.listdir(obsm_dir):
            if fn.endswith(".parquet"):
                key = fn[:-8]
                detected[f"obsm:{key}"] = os.path.join(obsm_dir, fn)
                adata.obsm[key] = pd.read_parquet(os.path.join(obsm_dir, fn)).to_numpy()

    varm_dir = _fname(in_dir, prefix, "varm")
    if os.path.isdir(varm_dir):
        for fn in os.listdir(varm_dir):
            if fn.endswith(".parquet"):
                key = fn[:-8]
                detected[f"varm:{key}"] = os.path.join(varm_dir, fn)
                adata.varm[key] = pd.read_parquet(os.path.join(varm_dir, fn)).to_numpy()

    # obsp / varp
    obsp_dir = _fname(in_dir, prefix, "obsp")
    if os.path.isdir(obsp_dir):
        keys = {fn.split("_csr_")[0] for fn in os.listdir(obsp_dir) if "_csr_" in fn}
        for fn in os.listdir(obsp_dir):
            if fn.endswith(".parquet") and "_csr_" not in fn:
                key = fn[:-8]
                detected[f"obsp:{key}"] = os.path.join(obsp_dir, fn)
                adata.obsp[key] = pd.read_parquet(os.path.join(obsp_dir, fn)).to_numpy()
        for key in keys:
            detected[f"obsp:{key}"] = os.path.join(obsp_dir, f"{key}_csr_*.parquet")
            adata.obsp[key] = _read_csr(os.path.join(obsp_dir, key))

    varp_dir = _fname(in_dir, prefix, "varp")
    if os.path.isdir(varp_dir):
        keys = {fn.split("_csr_")[0] for fn in os.listdir(varp_dir) if "_csr_" in fn}
        for fn in os.listdir(varp_dir):
            if fn.endswith(".parquet") and "_csr_" not in fn:
                key = fn[:-8]
                detected[f"varp:{key}"] = os.path.join(varp_dir, fn)
                adata.varp[key] = pd.read_parquet(os.path.join(varp_dir, fn)).to_numpy()
        for key in keys:
            detected[f"varp:{key}"] = os.path.join(varp_dir, f"{key}_csr_*.parquet")
            adata.varp[key] = _read_csr(os.path.join(varp_dir, key))

    # layers
    layers_dir = _fname(in_dir, prefix, "layers")
    if os.path.isdir(layers_dir):
        keys = {fn.split("_csr_")[0] for fn in os.listdir(layers_dir) if "_csr_" in fn}
        for fn in os.listdir(layers_dir):
            if fn.endswith(".parquet") and "_csr_" not in fn:
                key = fn[:-8]
                detected[f"layers:{key}"] = os.path.join(layers_dir, fn)
                adata.layers[key] = pd.read_parquet(os.path.join(layers_dir, fn)).to_numpy()
        for key in keys:
            detected[f"layers:{key}"] = os.path.join(layers_dir, f"{key}_csr_*.parquet")
            adata.layers[key] = _read_csr(os.path.join(layers_dir, key))

    # uns (best-effort)
    uns_path = _fname(in_dir, prefix, "uns.json")
    if os.path.exists(uns_path):
        detected["uns"] = uns_path
        with open(uns_path, "r", encoding="utf-8") as f:
            adata.uns.update(json.load(f))

    if verbose:
        for k in sorted(detected.keys()):
            print(f"detected {k} : {detected[k]}")

    return adata


def load_csr_only(in_dir: str, prefix: str = "") -> sparse.csr_matrix:
    """Convenience helper to load only X as CSR."""
    return _read_csr(_fname(in_dir, prefix, "X"))
