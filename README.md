# anndata-parquet-utils

Split AnnData into Parquet files (X/obs/var/layers/obsm/varm/obsp/varp) and restore it back.

## Install

From GitHub:

```bash
pip install git+https://github.com/kh22cho/anndata-parquet-utils.git
```

(Optional) Editable install for local development:

```bash
pip install -e .
```

## Usage

```python
import anndata as ad
from anndata_parquet_utils import to_parquet, from_parquet

# Save
# - prefix/suffix: optional (default: "")
# - obs_cols/var_cols: optional (default: all columns)
out_dir = "/path/to/save_dir"
to_parquet(
    adata,
    out_dir,
    prefix="group_",   # optional; prepended to top-level file names
    suffix="_v1",      # optional; appended to top-level file names
    obs_cols=["colA", "colB"],  # optional; save selected obs columns only
    var_cols=["colC", "colD"],  # optional; save selected var columns only
    layers_keys=["counts", "log1p"],  # optional; save selected layers only
    obsm_keys=["pca", "umap"],        # optional; save selected obsm only
    varm_keys=["pca_loadings"],       # optional; save selected varm only
)

# Load
# - prefix/suffix: optional (default: "")
# - verbose: optional (default: True) prints detected files
adata = from_parquet(
    out_dir,
    prefix="group_",
    suffix="_v1",
    verbose=True,
)
```

## Output layout (default)

```
{prefix}X{suffix}_csr.parquet
{prefix}obs{suffix}.parquet
{prefix}var{suffix}.parquet
{prefix}obsm{suffix}/{key}.parquet
{prefix}varm{suffix}/{key}.parquet
{prefix}obsp{suffix}/{key}_csr.parquet (or .parquet if dense)
{prefix}varp{suffix}/{key}_csr.parquet (or .parquet if dense)
{prefix}layers{suffix}/{key}_csr.parquet (or .parquet if dense)
{prefix}uns.json{suffix} (best-effort)
```

## Notes

- Sparse matrices are stored as a single CSR parquet with data/indices/indptr/shape.
- `obs_cols` / `var_cols` can be passed to save only selected columns.
- `layers_keys` / `obsm_keys` / `varm_keys` can be passed to save only selected parts.
- `uns` is saved to JSON with best-effort serialization.
- Parquet I/O requires `pyarrow`.

## License

MIT
