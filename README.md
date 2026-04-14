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
# prefix/suffix are optional; they are prepended/appended to top-level file names
out_dir = "/path/to/save_dir"
to_parquet(adata, out_dir, prefix="group_", suffix="_v1")

# Load
adata = from_parquet(out_dir, prefix="group_", suffix="_v1", verbose=True)
```

## Output layout (default)

```
{prefix}X{suffix}_csr_*.parquet
{prefix}obs{suffix}.parquet
{prefix}var{suffix}.parquet
{prefix}obsm{suffix}/{key}.parquet
{prefix}varm{suffix}/{key}.parquet
{prefix}obsp{suffix}/{key}_csr_*.parquet (or .parquet if dense)
{prefix}varp{suffix}/{key}_csr_*.parquet (or .parquet if dense)
{prefix}layers{suffix}/{key}_csr_*.parquet (or .parquet if dense)
{prefix}uns.json{suffix} (best-effort)
```

## Notes

- Sparse matrices are stored as CSR components: data/indices/indptr/shape.
- `obs_cols` / `var_cols` can be passed to save only selected columns.
- `uns` is saved to JSON with best-effort serialization.
- Parquet I/O requires `pyarrow`.

## License

MIT
