from pathlib import Path

# 使用/是因为方法被path库重写了
# 因为config文件和process在同一个目录下 所以可以使用__file__
ROOT_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"

SEQ_LEN = 128

BATCH_SIZE = 64
EMBEDDING_DIM = 300
HIDDEN_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 10
