from typing import List

import torch
from transformers import ClapModel, ClapProcessor

from audio_data_poisoning.models.base_model import BaseModel


class CLAP(BaseModel):

    def __init__(self, model_type: str = "laion/clap-htsat-unfused"):
        super().__init__(model=ClapModel, model_type=model_type)
        self.processor = ClapProcessor.from_pretrained(model_type)

    def get_features(self, texts: List[str], *args, **kwargs) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)

        with torch.no_grad():
            return self.model.get_text_features(**inputs)
