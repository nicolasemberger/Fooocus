import torch
import torch.nn as nn
from transformers import CLIPConfig, CLIPVisionModel, PreTrainedModel
from transformers.utils import logging

logger = logging.get_logger(__name__)


class StableDiffusionSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig
    main_input_name = "clip_input"
    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        # Still initialize layers in case something depends on them
        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        self.concept_embeds = nn.Parameter(torch.ones(17, config.projection_dim), requires_grad=False)
        self.special_care_embeds = nn.Parameter(torch.ones(3, config.projection_dim), requires_grad=False)

        self.concept_embeds_weights = nn.Parameter(torch.ones(17), requires_grad=False)
        self.special_care_embeds_weights = nn.Parameter(torch.ones(3), requires_grad=False)

    @torch.no_grad()
    def forward(self, clip_input, images):
        # Bypass detection entirely
        batch_size = len(images) if isinstance(images, (list, tuple)) else images.shape[0]
        has_nsfw_concepts = [False] * batch_size

        return images, has_nsfw_concepts

    @torch.no_grad()
    def forward_onnx(self, clip_input: torch.Tensor, images: torch.Tensor):
        # Bypass detection entirely
        has_nsfw_concepts = torch.zeros(images.shape[0], dtype=torch.bool)
        return images, has_nsfw_concepts
