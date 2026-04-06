import torch

from audio_data_poisoning.common.utils import DEVICE

from diffusers import AudioLDMPipeline

from audio_data_poisoning.models.base_model import BaseModel


class AudioLDM(BaseModel):

    def __init__(
        self,
        model_type: str = "cvssp/audioldm",
        torch_dtype: torch.dtype = torch.float16,
    ):
        super().__init__(
            model=AudioLDMPipeline,
            model_type=model_type,
            torch_dtype=torch_dtype,
        )

    def get_features(self, *args, **kwargs):
        raise NotImplementedError("AudioLDM does not support feature extraction.")
