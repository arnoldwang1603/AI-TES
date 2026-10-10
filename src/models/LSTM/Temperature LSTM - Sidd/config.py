from pathlib import Path

MODEL_VERSION = "PhysTFv3"

BASE_DIR       = Path(__file__).parent
DATA_ROOT      = BASE_DIR.parent / "data"
TRAIN_DATA_DIR = DATA_ROOT / "training_data" / "Latest Database (Use this for training)"
TEST_DATA_DIR  = DATA_ROOT / "tests" / "test_in_10s"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR    = BASE_DIR / "results"

SEQ_LEN = 1440
DT      = 10.0

SCALE_COLS = ["Time (s)", "T_outer (C)", "T_inner (C)", "T_avg (C)", "Input Temperature (C)"]
IDX_TIME   = 0
IDX_TOUTER = 1
IDX_TINNER = 2
IDX_TAVG   = 3
IDX_TIN    = 4

D_MODEL    = 128
N_HEADS    = 4
N_LAYERS   = 4
FFN_DIM    = 512
TF_DROPOUT = 0.20

D = 0.1016   # m  outer diameter
W = 0.00635  # m  wall thickness

X_INIT = 0.058
X_MIN  = 0.01
X_MAX  = D - 0.01

ALPHA1_MIN = 1e-7;  ALPHA1_MAX = 2e-5
ALPHA2_MIN = 5e-8;  ALPHA2_MAX = 2e-6
ALPHA3_MIN = 1e-7;  ALPHA3_MAX = 8e-7

LSTM_DIM      = 64
LSTM_LAYERS   = 2
N_ATTN_LAYERS = 2
ATTN_WINDOW   = 120   # ±120 steps = ±20 min

TA_DELTA_MAX = 300.0  # °C

BATCH_SIZE   = 16
MAX_EPOCHS   = 300
PATIENCE     = 60
SEED         = 42
VAL_SPLIT    = 0.20

LR           = 5e-4
WEIGHT_DECAY = 1e-3
GRAD_CLIP    = 1.0

T_INNER_WEIGHT     = 1.0
T_OUTER_WEIGHT     = 6.0
T_AVG_WEIGHT       = 3.0
PHYS_REG_WEIGHT    = 0.05
ALPHA_SMOOTH_WEIGHT = 0.005
SMOOTH_TA_WEIGHT   = 0.01

TAVG_WARMUP_START = 20
TAVG_WARMUP_LEN   = 20
