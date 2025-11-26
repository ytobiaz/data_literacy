from pathlib import Path
import os
from paths import *

# Define the project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#################################################################################

# Define subdirectories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
BLD_DIR = PROJECT_ROOT / "bld"
LOGS_DIR = PROJECT_ROOT / "logs"
BASELINES_DIR = PROJECT_ROOT / "baselines"
#################################################################################
# Raw data files
WORKING_BERLIN_DATA = DATA_DIR / "raw/berlin_data.csv"
WORKING_NEWYORK_DATA = DATA_DIR / "raw/ny_data.csv"  # only needed if you ever use taxi


#################################################################################
# Feature files
FEATURE_LIST_STRAVA = DATA_DIR / "berlin_features_list.json"

FEATURE_LIST_NEWYORK = DATA_DIR / "ny_feature_list.json"

#################################################################################
GNNUI_DATA_STRAVA = DATA_DIR / "processed/working_data_for_strava.pkl"
GNNUI_DATA_TAXI = DATA_DIR / "processed/working_data_for_taxi.pkl"

GNNUI_DATA_FACSIMILE_STORAGE = (
    DATA_DIR / "data_prepared"
)  
GNNUI_BASELINES = BLD_DIR / "baseline"

#################################################################################
# Graph information
NEW_YORK_GRAPH = DATA_DIR / "raw/ny_graph_geometry.parquet"
BERLIN_GRAPH = DATA_DIR / "raw/berlin_graph_geometry.parquet"

NEW_YORK_INVERTED_UNDIRECTED_GRAPH = DATA_DIR / "processed/ny_network_graph.pickle"
PROCESSED_BERLIN_CYCLING_NETWORK_GRAPH = (
    DATA_DIR / "processed/berlin_network_graph.pickle"
)

#################################################################################
# Adjacency matrices
BERLIN_ADJACENCY_BINARY = DATA_DIR / "berlin_adjacency_binary.parquet"

BERLIN_ADJACENCY_BIRD = DATA_DIR / "berlin_adjacency_birdfly.parquet"

BERLIN_ADJACENCY_TRAVELTIME = DATA_DIR / "berlin_adjacency_traveltime.parquet"

BERLIN_ADJACENCY_ROADDISTANCE = DATA_DIR / "berlin_adjacency_roaddistance.parquet"

BERLIN_ADJACENCY_SIMILARITY = DATA_DIR / "berlin_adjacency_similarity.parquet"

BERLIN_ADJACENCY_BINARY_AND_BIRDFLY = (
    DATA_DIR / "berlin_adjacency_binary_and_birdfly.parquet"
)

BERLIN_ADJACENCY_BINARY_AND_TRAVELTIME = (
    DATA_DIR / "berlin_adjacency_binary_and_traveltime.parquet"
)

BERLIN_ADJACENCY_BINARY_AND_ROADDISTANCE = (
    DATA_DIR / "berlin_adjacency_binary_and_roaddistance.parquet"
)

BERLIN_ADJACENCY_BINARY_AND_SIMILARITY = (
    DATA_DIR / "berlin_adjacency_binary_and_similarity.parquet"
)

BERLIN_ADJACENCY_SIMILARITY_AND_TRAVELTIME = (
    DATA_DIR / "berlin_adjacency_similarity_and_traveltime.parquet"
)

BERLIN_ADJACENCY_SIMILARITY_AND_ROADDISTANCE = (
    DATA_DIR / "berlin_adjacency_similarity_and_roaddistance.parquet"
)

BERLIN_ADJACENCY_SIMILARITY_AND_BIRDFLY = (
    DATA_DIR / "berlin_adjacency_similarity_and_birdfly.parquet"
)


NY_ADJACENCY_BINARY = DATA_DIR / "ny_adjacency_binary.parquet"


NY_ADJACENCY_BIRD = DATA_DIR / "ny_adjacency_birdfly.parquet"


NY_ADJACENCY_TRAVELTIME = DATA_DIR / "ny_adjacency_traveltime.parquet"


NY_ADJACENCY_ROADDISTANCE = DATA_DIR / "ny_adjacency_roaddistance.parquet"


NY_ADJACENCY_SIMILARITY = DATA_DIR / "ny_adjacency_similarity.parquet"


NY_ADJACENCY_BINARY_AND_BIRDFLY = DATA_DIR / "ny_adjacency_binary_and_birdfly.parquet"

NY_ADJACENCY_BINARY_AND_TRAVELTIME = (
    DATA_DIR / "ny_adjacency_binary_and_traveltime.parquet"
)


NY_ADJACENCY_BINARY_AND_ROADDISTANCE = (
    DATA_DIR / "ny_adjacency_binary_and_roaddistance.parquet"
)


NY_ADJACENCY_BINARY_AND_SIMILARITY = (
    DATA_DIR / "ny_adjacency_binary_and_similarity.parquet"
)


BERLIN_ADJACENCY_SIMILARITY_AND_TRAVELTIME_NY = (
    DATA_DIR / "ny_adjacency_similarity_and_traveltime.parquet"
)


BERLIN_ADJACENCY_SIMILARITY_AND_ROADDISTANCE_NY = (
    DATA_DIR / "ny_adjacency_similarity_and_roaddistance.parquet"
)


BERLIN_ADJACENCY_SIMILARITY_AND_BIRDFLY_NY = (
    DATA_DIR / "ny_adjacency_similarity_and_birdfly.parquet"
)

##################################################################################
BERLIN_MATRIX_TRAVELTIME = DATA_DIR / "BERLIN_MATRIX_TRAVELTIME.pkl"
BERLIN_MATRIX_TRAVELDISTANCE = DATA_DIR / "BERLIN_MATRIX_TRAVELDISTANCE.pkl"
NY_MATRIX_TRAVELTIME= DATA_DIR / "NY_MATRIX_TRAVELTIME.pkl"
NY_MATRIX_TRAVELDISTANCE = DATA_DIR / "NY_MATRIX_TRAVELDISTANCE.pkl"

##################################################################################
MODEL_RESULTS = BLD_DIR / "model"


# Allow environment-based overrides
def get_env_path(env_var, default_path):
    return Path(os.getenv(env_var, default_path))


# Example: Override paths using environment variables
CUSTOM_DATA_DIR = get_env_path("CUSTOM_DATA_DIR", DATA_DIR)
