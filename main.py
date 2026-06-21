from src.data.extract import extract_dataset
from src.data.convert_to_csv import convert_all
from src.data.clean import clean_all
from src.graphs.build_graphs import build_all_graphs
from src.graphs.report_graphs import generate_graph_reports
from src.graphs.user_metrics import generate_user_metrics
from src.analysis.homophily import run_homophily
from src.data.split import split_user_interactions


def main() -> None:
    extract_dataset()
    convert_all()
    clean_all()
    build_all_graphs()
    generate_graph_reports()
    generate_user_metrics()
    run_homophily(data_source="full")
    run_homophily(data_source="train")
    split_user_interactions()


if __name__ == "__main__":
    main()