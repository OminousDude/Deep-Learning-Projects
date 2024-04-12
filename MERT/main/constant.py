from enum import Enum

class Constants(Enum):
    BATCH_SIZE = 128
    MAX_SEQ_LEN = 128
    VOCAB_SIZE = 30000
    NUM_HEADS = 12
    POS_ENC_LEN = MAX_SEQ_LEN
    EMB_DIM = 768
    FEED_FORWARD_DIM = EMB_DIM * 4
    NUM_LAYERS = 12
    DEVICE = "cuda"
    RANDOM_SEED = 42