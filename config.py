# -----------------------
# Training Settings
# -----------------------

TRAIN_ROOT = "/data/ahsoka/eocp/wengler/height_database/spline/train"
VAL_ROOT   = "/data/ahsoka/eocp/wengler/height_database/spline/val"

TRAIN_CACHE_ROOT = "/data/ahsoka/eocp/wengler/height_database/spline/npz_canopygap/train"
VAL_CACHE_ROOT = "/data/ahsoka/eocp/wengler/height_database/spline/npz_canopygap/val"

BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 1e-4

NUM_BANDS = 220
S1_BANDS = 3

DEVICE = "cuda"

NUM_WORKERS = 4
NUM_THREADS = 4
AUGMENT = True
TRAIN_SAMPLES_PER_EPOCH = 1000
VAL_EVERY = 3



MODEL_OUT = "/data/ahsoka/eocp/wengler/ground_fractions/canopygap_net/models/output/spline_prestem_08062026.pth"
LOG_PATH = "/data/ahsoka/eocp/wengler/ground_fractions/canopygap_net/models/log/spline_prestem_08062026.log"
TB_LOG_DIR = "/data/ahsoka/eocp/wengler/ground_fractions/canopygap_net/models/log/TB"


# -----------------------
# Prediction Settings
# -----------------------

PREDICTION_SPLINE_ROOT = "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/ThermSpline_DC"
PREDICTION_S1_ROOT = "/data/ahsoka/eocp/wengler/height_database/composite/S1/RLP/median_3035"

PREDICTION_OUTPUT = "/data/ahsoka/eocp/wengler/ground_fractions/out/RLP_2020_spline_08062026.tif"
PREDICTION_OUTPUT_ROOT = "//data/ahsoka/eocp/wengler/ground_fractions/out/tiles_20"

PREDICTION_TILE_LIST_FILE = "/data/ahsoka/eocp/wengler/height_database/composite/rlp_tileallow/Tile_allow_spline.txt"
PREDICTION_YEARS = [2020]

PREDICTION_PATCH_SIZE = 256
PREDICTION_BATCH_SIZE = 8
PREDICTION_TILE_SIZE = 1024

PREDICTION_MODEL = MODEL_OUT
