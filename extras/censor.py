import os
import numpy as np
import torch

from transformers import CLIPConfig, CLIPImageProcessor

# These imports are kept for compatibility even if unused now
import ldm_patched.modules.model_management as model_management
import modules.config
from extras.safety_checker.models.safety_checker import StableDiffusionSafetyChecker
from ldm_patched.modules.model_patcher import ModelPatcher

safety_checker_repo_root = os.path.join(os.path.dirname(__file__), 'safety_checker')
config_path = os.path.join(safety_checker_repo_root, "configs", "config.json")
preprocessor_config_path = os.path.join(safety_checker_repo_root, "configs", "preprocessor_config.json")


class Censor:
    def __init__(self):
        # Kept for compatibility, but unused now
        self.safety_checker_model: ModelPatcher | None = None
        self.clip_image_processor: CLIPImageProcessor | None = None
        self.load_device = torch.device('cpu')
        self.offload_device = torch.device('cpu')

    def init(self):
        # Safety checker initialization is disabled
        pass

    def censor(self, images: list | np.ndarray) -> list | np.ndarray:
        # Bypasses all NSFW detection and returns original images

        single = False
        if not isinstance(images, (list, np.ndarray)):
            images = [images]
            single = True

        # Cast to uint8 to keep output format compatible
        checked_images = [image.astype(np.uint8) for image in images]

        if single:
            checked_images = checked_images[0]

        return checked_images


# You can now use this to override censorship globally
default_censor = Censor().censor
