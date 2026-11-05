from Framework import MethodError
from Methods.HTGS.Model import HTGSModel
from Methods.HTGS.Trainer import HTGSTrainer
from Methods.VRHTGS.Renderer import VRHTGSRenderer

try:
    from Methods.HTGS.Model import SHEnvironmentMap
except ImportError:
    raise MethodError("VRHTGS requires the `vrhtgs-training` branch of the HTGS method. Please ensure you cloned the "
                      "correct branch of the repository using `git clone -b vrhtgs-training git@github.com:nerficg-project/HTGS.git`")

MODEL = HTGSModel
RENDERER = VRHTGSRenderer
TRAINING_INSTANCE = HTGSTrainer
