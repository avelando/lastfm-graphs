from src.lastfm_project.data.extract import extract_dataset
from src.lastfm_project.data.convert_to_csv import convert_all
from src.lastfm_project.data.clean import clean_all
from src.lastfm_project.graphs.build_graphs import build_all_graphs
from src.lastfm_project.graphs.report_graphs import generate_graph_reports


def main() -> None:
    extract_dataset()
    convert_all()
    clean_all()
    build_all_graphs()
    generate_graph_reports()


if __name__ == "__main__":
    main()