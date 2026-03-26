import torch
from diffusers import AudioLDMPipeline

from wimudp.data_poisoning.nightshade.vocoder import Vocoder

MODEL_TYPE = "cvssp/audioldm"


class Pipeline:
    def __init__(self):
        model = AudioLDMPipeline.from_pretrained(MODEL_TYPE, torch_dtype=torch.float16)
        self.model: AudioLDMPipeline = model.to("cuda")

    def get_latent(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.half()

        return self.model.vae.encode(tensor.unsqueeze(0)).latent_dist.mean
