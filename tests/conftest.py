import pytest
import numpy as np
import anndata as ad
import pandas as pd
import scipy.sparse


@pytest.fixture
def synthetic_adata():
    """Create a synthetic AnnData for testing."""
    np.random.seed(42)
    n_cells = 200
    n_genes = 100

    X = scipy.sparse.csr_matrix(np.random.poisson(1.0, (n_cells, n_genes)))

    donor_ids = [f"donor_{i % 4}" for i in range(n_cells)]
    disease_status_raw = [i % 2 for i in range(n_cells)]
    donor_label = {}
    for d, lbl in zip(donor_ids, disease_status_raw):
        donor_label[d] = lbl % 2
    disease_status = [donor_label[d] for d in donor_ids]

    cell_types = [f"type_{i % 3}" for i in range(n_cells)]

    obs = pd.DataFrame(
        {
            "donor_id": donor_ids,
            "disease_status": disease_status,
            "cell_type": cell_types,
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["X_scVI"] = np.random.randn(n_cells, 30).astype(np.float32)

    return adata
