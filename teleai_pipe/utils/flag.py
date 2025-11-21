import importlib
from loguru import logger
import importlib

# 无损
FLASH_ATTN_3_AVAILABLE = False
FLASH_ATTN_2_AVAILABLE = False
XFORMERS_AVAILABLE = False
SDPA_AVAILABLE = False
# 有损
SAGE_ATTN_AVAILABLE = False
SPARGE_ATTN_AVAILABLE = False


try:
    FLASH_ATTN_3_AVAILABLE = (
        importlib.util.find_spec("flash_attn_interface") is not None
    )
    logger.info(f"Flash attention 3 is {FLASH_ATTN_3_AVAILABLE}")
except ModuleNotFoundError:
    logger.info("Flash attention 3 is not available")

try:
    FLASH_ATTN_2_AVAILABLE = importlib.util.find_spec("flash_attn") is not None
    logger.info(f"Flash attention 2 is {FLASH_ATTN_2_AVAILABLE}")
except ModuleNotFoundError:
    logger.info("Flash attention 2 is not available")

try:
    XFORMERS_AVAILABLE = importlib.util.find_spec("xformers") is not None
    logger.info(f"XFormers is {XFORMERS_AVAILABLE}")
except ModuleNotFoundError:
    logger.info("XFormers is not available")

try:
    from torch.nn.functional import scaled_dot_product_attention

    logger.info("Torch SDPA is available")
    SDPA_AVAILABLE = True
except ModuleNotFoundError:
    logger.info("Torch SDPA is not available")


try:
    SAGE_ATTN_AVAILABLE = (
        importlib.util.find_spec("tele_fuser.sageattention") is not None
    )
    logger.info(f"Sage attention is {SAGE_ATTN_AVAILABLE}")
except ModuleNotFoundError:
    logger.info("Sage attention is not available")

try:
    SPARGE_ATTN_AVAILABLE = (
        importlib.util.find_spec("tele_fuser.spas_sage_attn") is not None
    )
    logger.info(f"Sparge attention is {SPARGE_ATTN_AVAILABLE}")
except ModuleNotFoundError:
    logger.info("Sparge attention is not available")
