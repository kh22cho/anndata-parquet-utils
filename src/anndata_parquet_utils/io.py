"""AnnData <-> Parquet split I/O utilities.

Layout (prefix optional):
- {prefix}X_csr_data_indices.parquet
- {prefix}X_csr_indptr_shape.parquet
- {prefix}obs.parquet
- {prefix}var.parquet
- {prefix}obsm/{key}.parquet
- {prefix}varm/{key}.parquet
- {prefix}obsp/{key}_csr_data_indices.parquet (or .parquet if dense)
- {prefix}obsp/{key}_csr_indptr_shape.parquet
- {prefix}varp/{key}_csr_data_indices.parquet (or .parquet if dense)
- {prefix}varp/{key}_csr_indptr_shape.parquet
- {prefix}layers/{key}_csr_data_indices.parquet (or .parquet if dense)
- {prefix}layers/{key}_csr_indptr_shape.parquet
- {prefix}uns.json (best-effort)
"""

from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

import anndata as ad


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _fname(dir_path: str, prefix: str, name: str, suffix: str) -> str:
    return os.path.join(dir_path, f"{prefix}{name}{suffix}")


def _pq_write_table(table: pa.Table, path: str, compression: str | None) -> None:
    # pyarrow default compression is typically snappy; avoid overriding it when None.
    if compression is None:
        pq.write_table(table, path)
    else:
        pq.write_table(table, path, compression=compression)


def _write_csr_all(
    X: sparse.csr_matrix, base_path: str, compression: str | None = None
) -> None:
    _pq_write_table(
        pa.table({"data": pa.array(X.data)}),
        base_path + "_csr_data.parquet",
        compression=compression,
    )
    _pq_write_table(
        pa.table({"indices": pa.array(X.indices)}),
        base_path + "_csr_indices.parquet",
        compression=compression,
    )
    _pq_write_table(
        pa.table({"indptr": pa.array(X.indptr)}),
        base_path + "_csr_indptr.parquet",
        compression=compression,
    )
    _pq_write_table(
        pa.table({"shape0": [X.shape[0]], "shape1": [X.shape[1]]}),
        base_path + "_csr_shape.parquet",
        compression=compression,
    )


def _write_csr_two(
    X: sparse.csr_matrix, base_path: str, compression: str | None = None
) -> None:
    _pq_write_table(
        pa.table({"data": pa.array(X.data), "indices": pa.array(X.indices)}),
        base_path + "_csr_data_indices.parquet",
        compression=compression,
    )

    indptr_tbl = pa.table({"indptr": pa.array(X.indptr)})
    indptr_tbl = indptr_tbl.replace_schema_metadata(
        {
            b"format": b"csr",
            b"shape0": str(X.shape[0]).encode("ascii"),
            b"shape1": str(X.shape[1]).encode("ascii"),
        }
    )
    _pq_write_table(
        indptr_tbl,
        base_path + "_csr_indptr_shape.parquet",
        compression=compression,
    )


def _read_csr(base_path: str) -> sparse.csr_matrix:
    data_path = base_path + "_csr_data_indices.parquet"
    indptr_path = base_path + "_csr_indptr_shape.parquet"
    if os.path.exists(data_path) and os.path.exists(indptr_path):
        t_data = pq.read_table(data_path)
        t_indptr = pq.read_table(indptr_path)

        # Avoid an extra combine_chunks() copy; ChunkedArray.to_numpy() concatenates as needed.
        data = t_data.column("data").to_numpy(zero_copy_only=False)
        indices = t_data.column("indices").to_numpy(zero_copy_only=False)
        indptr = t_indptr.column("indptr").to_numpy(zero_copy_only=False)

        md = t_indptr.schema.metadata or {}
        if b"shape0" in md and b"shape1" in md:
            shape0 = int(md[b"shape0"].decode("ascii"))
            shape1 = int(md[b"shape1"].decode("ascii"))
        elif "shape0" in t_indptr.column_names and "shape1" in t_indptr.column_names:
            shape0 = int(t_indptr.column("shape0")[0].as_py())
            shape1 = int(t_indptr.column("shape1")[0].as_py())
        else:
            raise ValueError(f"Missing shape metadata for {indptr_path}")

        return sparse.csr_matrix((data, indices, indptr), shape=(shape0, shape1))

    # Legacy single-file CSR
    csr_path = base_path + "_csr.parquet"
    if os.path.exists(csr_path):
        df = pd.read_parquet(csr_path)
        data = np.asarray(df["data"].iloc[0])
        indices = np.asarray(df["indices"].iloc[0])
        indptr = np.asarray(df["indptr"].iloc[0])
        shape = (int(df["shape0"].iloc[0]), int(df["shape1"].iloc[0]))
        return sparse.csr_matrix((data, indices, indptr), shape=shape)

    # Legacy split files
    data = pd.read_parquet(base_path + "_csr_data.parquet")["data"].to_numpy()
    indices = pd.read_parquet(base_path + "_csr_indices.parquet")["indices"].to_numpy()
    indptr = pd.read_parquet(base_path + "_csr_indptr.parquet")["indptr"].to_numpy()
    shape_df = pd.read_parquet(base_path + "_csr_shape.parquet")
    shape = (int(shape_df["shape0"].iloc[0]), int(shape_df["shape1"].iloc[0]))
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


def _write_matrix(
    value,
    base_path: str,
    compression: str | None = None,
    x_split: str = "two",
) -> None:
    if sparse.issparse(value):
        if x_split == "all":
            _write_csr_all(value.tocsr(), base_path, compression=compression)
        elif x_split == "two":
            _write_csr_two(value.tocsr(), base_path, compression=compression)
        else:
            raise ValueError("x_split must be 'all' or 'two'")
    else:
        table = pa.Table.from_pandas(pd.DataFrame(value), preserve_index=False)
        _pq_write_table(table, base_path + ".parquet", compression=compression)


def _read_matrix(base_path: str):
    if os.path.exists(base_path + "_csr_data_indices.parquet"):
        return _read_csr(base_path)
    csr_path = base_path + "_csr.parquet"
    if os.path.exists(csr_path):
        return _read_csr(base_path)
    if os.path.exists(base_path + "_csr_data.parquet"):
        return _read_csr(base_path)
    return pd.read_parquet(base_path + ".parquet").to_numpy()


def _select_keys(container, keys, label: str):
    if keys is None:
        return list(container.keys())
    missing = [k for k in keys if k not in container]
    if missing:
        raise KeyError(f"Missing {label} keys: {missing}")
    return list(keys)


def to_parquet(
    adata: ad.AnnData,
    out_dir: str,
    prefix: str = "",
    suffix: str = "",
    obs_cols=None,
    var_cols=None,
    layers_keys=None,
    obsm_keys=None,
    varm_keys=None,
    compression: str | None = None,
    x_split: str = "two",
) -> None:
    """Save AnnData to a split parquet directory.

    prefix: optional string prepended to top-level file names (e.g. "group_").
    suffix: optional string appended to top-level file names (e.g. "_v1").
    obs_cols/var_cols: optional column lists to select before saving.
    layers_keys/obsm_keys/varm_keys: optional key lists to select before saving.
    x_split: "two" (default) or "all" for CSR split strategy.
    """
    _ensure_dir(out_dir)

    # X
    if sparse.issparse(adata.X):
        if x_split == "all":
            _write_csr_all(
                adata.X.tocsr(),
                _fname(out_dir, prefix, "X", suffix),
                compression=compression,
            )
        elif x_split == "two":
            _write_csr_two(
                adata.X.tocsr(),
                _fname(out_dir, prefix, "X", suffix),
                compression=compression,
            )
        else:
            raise ValueError("x_split must be 'all' or 'two'")
    else:
        table = pa.Table.from_pandas(pd.DataFrame(adata.X), preserve_index=False)
        _pq_write_table(
            table, _fname(out_dir, prefix, "X", suffix) + ".parquet", compression=compression
        )

    # obs / var
    obs_df = adata.obs if obs_cols is None else adata.obs.loc[:, obs_cols]
    var_df = adata.var if var_cols is None else adata.var.loc[:, var_cols]
    _pq_write_table(
        pa.Table.from_pandas(obs_df, preserve_index=True),
        _fname(out_dir, prefix, "obs", suffix) + ".parquet",
        compression=compression,
    )
    _pq_write_table(
        pa.Table.from_pandas(var_df, preserve_index=True),
        _fname(out_dir, prefix, "var", suffix) + ".parquet",
        compression=compression,
    )

    # obsm / varm
    if len(adata.obsm):
        obsm_dir = _fname(out_dir, prefix, "obsm", suffix)
        _ensure_dir(obsm_dir)
        for k in _select_keys(adata.obsm, obsm_keys, "obsm"):
            table = pa.Table.from_pandas(pd.DataFrame(adata.obsm[k]), preserve_index=False)
            _pq_write_table(
                table, os.path.join(obsm_dir, f"{k}.parquet"), compression=compression
            )

    if len(adata.varm):
        varm_dir = _fname(out_dir, prefix, "varm", suffix)
        _ensure_dir(varm_dir)
        for k in _select_keys(adata.varm, varm_keys, "varm"):
            table = pa.Table.from_pandas(pd.DataFrame(adata.varm[k]), preserve_index=False)
            _pq_write_table(
                table, os.path.join(varm_dir, f"{k}.parquet"), compression=compression
            )

    # obsp / varp
    if len(adata.obsp):
        obsp_dir = _fname(out_dir, prefix, "obsp", suffix)
        _ensure_dir(obsp_dir)
        for k, v in adata.obsp.items():
            _write_matrix(
                v, os.path.join(obsp_dir, k), compression=compression, x_split=x_split
            )

    if len(adata.varp):
        varp_dir = _fname(out_dir, prefix, "varp", suffix)
        _ensure_dir(varp_dir)
        for k, v in adata.varp.items():
            _write_matrix(
                v, os.path.join(varp_dir, k), compression=compression, x_split=x_split
            )

    # layers
    if len(adata.layers):
        layers_dir = _fname(out_dir, prefix, "layers", suffix)
        _ensure_dir(layers_dir)
        for k in _select_keys(adata.layers, layers_keys, "layers"):
            _write_matrix(
                adata.layers[k],
                os.path.join(layers_dir, k),
                compression=compression,
                x_split=x_split,
            )

    # uns (best-effort)
    if len(adata.uns):
        uns_path = _fname(out_dir, prefix, "uns.json", suffix)
        with open(uns_path, "w", encoding="utf-8") as f:
            json.dump(adata.uns, f, ensure_ascii=True, default=str)


def from_parquet(
    in_dir: str, prefix: str = "", suffix: str = "", verbose: bool = True
) -> ad.AnnData:
    """Load AnnData from a split parquet directory.

    prefix: optional string prepended to top-level file names (e.g. "group_").
    suffix: optional string appended to top-level file names (e.g. "_v1").
    """
    detected: Dict[str, str] = {}

    # X
    x_base = _fname(in_dir, prefix, "X", suffix)
    if os.path.exists(x_base + "_csr_data_indices.parquet") and os.path.exists(
        x_base + "_csr_indptr_shape.parquet"
    ):
        detected["X"] = (
            x_base + "_csr_data_indices.parquet, " + x_base + "_csr_indptr_shape.parquet"
        )
        X = _read_csr(x_base)
    elif os.path.exists(x_base + "_csr.parquet"):
        detected["X"] = x_base + "_csr.parquet (legacy)"
        X = _read_csr(x_base)
    elif os.path.exists(x_base + "_csr_data.parquet"):
        detected["X"] = x_base + "_csr_*.parquet (legacy)"
        X = _read_csr(x_base)
    else:
        detected["X"] = x_base + ".parquet"
        X = pd.read_parquet(x_base + ".parquet").to_numpy()

    obs_path = _fname(in_dir, prefix, "obs", suffix) + ".parquet"
    var_path = _fname(in_dir, prefix, "var", suffix) + ".parquet"
    detected["obs"] = obs_path
    detected["var"] = var_path
    obs = pd.read_parquet(obs_path)
    var = pd.read_parquet(var_path)

    adata = ad.AnnData(X=X, obs=obs, var=var)

    # obsm / varm
    obsm_dir = _fname(in_dir, prefix, "obsm", suffix)
    if os.path.isdir(obsm_dir):
        for fn in os.listdir(obsm_dir):
            if fn.endswith(".parquet"):
                key = fn[:-8]
                detected[f"obsm:{key}"] = os.path.join(obsm_dir, fn)
                adata.obsm[key] = pd.read_parquet(os.path.join(obsm_dir, fn)).to_numpy()

    varm_dir = _fname(in_dir, prefix, "varm", suffix)
    if os.path.isdir(varm_dir):
        for fn in os.listdir(varm_dir):
            if fn.endswith(".parquet"):
                key = fn[:-8]
                detected[f"varm:{key}"] = os.path.join(varm_dir, fn)
                adata.varm[key] = pd.read_parquet(os.path.join(varm_dir, fn)).to_numpy()

    # obsp / varp
    obsp_dir = _fname(in_dir, prefix, "obsp", suffix)
    if os.path.isdir(obsp_dir):
        keys = set()
        for fn in os.listdir(obsp_dir):
            if fn.endswith("_csr_data_indices.parquet"):
                keys.add(fn[: -len("_csr_data_indices.parquet")])
            elif fn.endswith("_csr_indptr_shape.parquet"):
                keys.add(fn[: -len("_csr_indptr_shape.parquet")])
            elif fn.endswith("_csr.parquet"):
                keys.add(fn[: -len("_csr.parquet")])
            elif "_csr_" in fn:
                keys.add(fn.split("_csr_")[0])
        for fn in os.listdir(obsp_dir):
            if (
                fn.endswith(".parquet")
                and "_csr_" not in fn
                and not fn.endswith("_csr.parquet")
                and not fn.endswith("_csr_data_indices.parquet")
                and not fn.endswith("_csr_indptr_shape.parquet")
            ):
                key = fn[:-8]
                detected[f"obsp:{key}"] = os.path.join(obsp_dir, fn)
                adata.obsp[key] = pd.read_parquet(os.path.join(obsp_dir, fn)).to_numpy()
        for key in keys:
            if os.path.exists(os.path.join(obsp_dir, f"{key}_csr_data_indices.parquet")) and os.path.exists(
                os.path.join(obsp_dir, f"{key}_csr_indptr_shape.parquet")
            ):
                detected[f"obsp:{key}"] = (
                    os.path.join(obsp_dir, f"{key}_csr_data_indices.parquet")
                    + ", "
                    + os.path.join(obsp_dir, f"{key}_csr_indptr_shape.parquet")
                )
            elif os.path.exists(os.path.join(obsp_dir, f"{key}_csr.parquet")):
                detected[f"obsp:{key}"] = (
                    os.path.join(obsp_dir, f"{key}_csr.parquet") + " (legacy)"
                )
            else:
                detected[f"obsp:{key}"] = os.path.join(obsp_dir, f"{key}_csr_*.parquet (legacy)")
            adata.obsp[key] = _read_csr(os.path.join(obsp_dir, key))

    varp_dir = _fname(in_dir, prefix, "varp", suffix)
    if os.path.isdir(varp_dir):
        keys = set()
        for fn in os.listdir(varp_dir):
            if fn.endswith("_csr_data_indices.parquet"):
                keys.add(fn[: -len("_csr_data_indices.parquet")])
            elif fn.endswith("_csr_indptr_shape.parquet"):
                keys.add(fn[: -len("_csr_indptr_shape.parquet")])
            elif fn.endswith("_csr.parquet"):
                keys.add(fn[: -len("_csr.parquet")])
            elif "_csr_" in fn:
                keys.add(fn.split("_csr_")[0])
        for fn in os.listdir(varp_dir):
            if (
                fn.endswith(".parquet")
                and "_csr_" not in fn
                and not fn.endswith("_csr.parquet")
                and not fn.endswith("_csr_data_indices.parquet")
                and not fn.endswith("_csr_indptr_shape.parquet")
            ):
                key = fn[:-8]
                detected[f"varp:{key}"] = os.path.join(varp_dir, fn)
                adata.varp[key] = pd.read_parquet(os.path.join(varp_dir, fn)).to_numpy()
        for key in keys:
            if os.path.exists(os.path.join(varp_dir, f"{key}_csr_data_indices.parquet")) and os.path.exists(
                os.path.join(varp_dir, f"{key}_csr_indptr_shape.parquet")
            ):
                detected[f"varp:{key}"] = (
                    os.path.join(varp_dir, f"{key}_csr_data_indices.parquet")
                    + ", "
                    + os.path.join(varp_dir, f"{key}_csr_indptr_shape.parquet")
                )
            elif os.path.exists(os.path.join(varp_dir, f"{key}_csr.parquet")):
                detected[f"varp:{key}"] = (
                    os.path.join(varp_dir, f"{key}_csr.parquet") + " (legacy)"
                )
            else:
                detected[f"varp:{key}"] = os.path.join(varp_dir, f"{key}_csr_*.parquet (legacy)")
            adata.varp[key] = _read_csr(os.path.join(varp_dir, key))

    # layers
    layers_dir = _fname(in_dir, prefix, "layers", suffix)
    if os.path.isdir(layers_dir):
        keys = set()
        for fn in os.listdir(layers_dir):
            if fn.endswith("_csr_data_indices.parquet"):
                keys.add(fn[: -len("_csr_data_indices.parquet")])
            elif fn.endswith("_csr_indptr_shape.parquet"):
                keys.add(fn[: -len("_csr_indptr_shape.parquet")])
            elif fn.endswith("_csr.parquet"):
                keys.add(fn[: -len("_csr.parquet")])
            elif "_csr_" in fn:
                keys.add(fn.split("_csr_")[0])
        for fn in os.listdir(layers_dir):
            if (
                fn.endswith(".parquet")
                and "_csr_" not in fn
                and not fn.endswith("_csr.parquet")
                and not fn.endswith("_csr_data_indices.parquet")
                and not fn.endswith("_csr_indptr_shape.parquet")
            ):
                key = fn[:-8]
                detected[f"layers:{key}"] = os.path.join(layers_dir, fn)
                adata.layers[key] = pd.read_parquet(os.path.join(layers_dir, fn)).to_numpy()
        for key in keys:
            if os.path.exists(os.path.join(layers_dir, f"{key}_csr_data_indices.parquet")) and os.path.exists(
                os.path.join(layers_dir, f"{key}_csr_indptr_shape.parquet")
            ):
                detected[f"layers:{key}"] = (
                    os.path.join(layers_dir, f"{key}_csr_data_indices.parquet")
                    + ", "
                    + os.path.join(layers_dir, f"{key}_csr_indptr_shape.parquet")
                )
            elif os.path.exists(os.path.join(layers_dir, f"{key}_csr.parquet")):
                detected[f"layers:{key}"] = (
                    os.path.join(layers_dir, f"{key}_csr.parquet") + " (legacy)"
                )
            else:
                detected[f"layers:{key}"] = os.path.join(layers_dir, f"{key}_csr_*.parquet (legacy)")
            adata.layers[key] = _read_csr(os.path.join(layers_dir, key))

    # uns (best-effort)
    uns_path = _fname(in_dir, prefix, "uns.json", suffix)
    if os.path.exists(uns_path):
        detected["uns"] = uns_path
        with open(uns_path, "r", encoding="utf-8") as f:
            adata.uns.update(json.load(f))

    if verbose:
        for k in sorted(detected.keys()):
            print(f"detected {k} : {detected[k]}")

    return adata


def load_csr_only(in_dir: str, prefix: str = "", suffix: str = "") -> sparse.csr_matrix:
    """Convenience helper to load only X as CSR."""
    return _read_csr(_fname(in_dir, prefix, "X", suffix))
