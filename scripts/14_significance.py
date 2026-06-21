from src.models.significance import run_significance


if __name__ == "__main__":
    # Baselines vs. baselines
    run_significance(model_a="userknn",   model_b="itemknn",               split="test")
    run_significance(model_a="userknn",   model_b="ease",                  split="test")
    run_significance(model_a="itemknn",   model_b="ease",                  split="test")

    # Score-based hybrids vs. best baseline
    run_significance(model_a="userknn",   model_b="knn_pop_ppr",           split="test")
    run_significance(model_a="itemknn",   model_b="itemknn_ppr",           split="test")
    run_significance(model_a="userknn",   model_b="itemknn_ppr",           split="test")
    run_significance(model_a="userknn",   model_b="userknn_itemknn_ppr",   split="test")
    run_significance(model_a="itemknn",   model_b="userknn_itemknn_ppr",   split="test")
    run_significance(model_a="ease",      model_b="userknn_itemknn_ppr",   split="test")

    # RRF hybrids vs. best baseline
    run_significance(model_a="userknn",   model_b="rrf_userknn_ppr",       split="test")
    run_significance(model_a="itemknn",   model_b="rrf_itemknn_ppr",       split="test")
    run_significance(model_a="userknn",   model_b="rrf_userknn_itemknn_ppr", split="test")
    run_significance(model_a="ease",      model_b="rrf_userknn_itemknn_ppr", split="test")

    # Score-based vs. RRF (mesma combinação)
    run_significance(model_a="knn_pop_ppr",         model_b="rrf_userknn_ppr",         split="test")
    run_significance(model_a="itemknn_ppr",         model_b="rrf_itemknn_ppr",         split="test")
    run_significance(model_a="userknn_itemknn_ppr", model_b="rrf_userknn_itemknn_ppr", split="test")
