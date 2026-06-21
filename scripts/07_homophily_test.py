from src.analysis.homophily import run_homophily


if __name__ == "__main__":
    run_homophily(data_source="full")
    run_homophily(data_source="train")
