# -----------------------
# Training Settings
# -----------------------

TRAIN_ROOT = "/data/ahsoka/eocp/wengler/height_database/DE_chips/train"
VAL_ROOT   = "/data/ahsoka/eocp/wengler/height_database/DE_chips/val"

LABEL_SUBDIR = "canopygap"


BATCH_SIZE = 12
EPOCHS = 100
LEARNING_RATE = 1e-4

NUM_BANDS = 220
S1_BANDS = 3

DEVICE = "cuda"

NUM_WORKERS = 2
NUM_THREADS = 8
AUGMENT = False
TRAIN_CHIPS_PER_EPOCH = 8000
VALIDATION_CHIPS = 2000

TRAIN_SEED = 42
VALIDATION_SEED = 42
VAL_EVERY = 1



MODEL_OUT = "/data/ahsoka/eocp/wengler/canopy_cover/canopygap_net/models/output/DE_model_test1_18092026.pth"
LOG_PATH = "/data/ahsoka/eocp/wengler/canopy_cover/canopygap_net/models/log/DE_model_test1_18092026.log"
TB_LOG_DIR = "/data/ahsoka/eocp/wengler/canopy_cover/canopygap_net/models/log/TB"


# -----------------------
# Prediction Settings
# -----------------------

PREDICTION_SPLINE_ROOT = "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/ThermSpline_DC"
PREDICTION_S1_ROOT = "/data/ahsoka/eocp/wengler/height_database/composite/S1/DE"


PREDICTION_OUTPUT_ROOT = "/data/ahsoka/eocp/wengler/canopy_cover/out/DE"
PREDICTION_OUTPUT_FILENAME_TEMPLATE = "{year}_canopycover_8m.tif"
S1_NODATA = -32768.0

PREDICTION_TILE_LIST_FILE = "/data/ahsoka/eocp/wengler/height_database/DE_out/Tile_allow_ger.txt"
PREDICTION_YEARS = [2021]

PREDICTION_PATCH_SIZE = 256
PREDICTION_BATCH_SIZE = 12 
PREDICTION_TILE_SIZE = 1024 #1536

PREDICTION_MODEL = MODEL_OUT
