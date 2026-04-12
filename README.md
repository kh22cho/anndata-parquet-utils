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
# prefix is optional; it is prepended to top-level file names
out_dir = "/path/to/save_dir"
to_parquet(adata, out_dir, prefix="group_")

# Load
adata2 = from_parquet(out_dir, prefix="group_", verbose=True)
```

## Output layout (default)

```
{prefix}X_csr_*.parquet
{prefix}obs.parquet
{prefix}var.parquet
{prefix}obsm/{key}.parquet
{prefix}varm/{key}.parquet
{prefix}obsp/{key}_csr_*.parquet (or .parquet if dense)
{prefix}varp/{key}_csr_*.parquet (or .parquet if dense)
{prefix}layers/{key}_csr_*.parquet (or .parquet if dense)
{prefix}uns.json (best-effort)
```

## Notes

- Sparse matrices are stored as CSR components: data/indices/indptr/shape.
- `uns` is saved to JSON with best-effort serialization.
- Parquet I/O requires `pyarrow`.

## License

MIT
