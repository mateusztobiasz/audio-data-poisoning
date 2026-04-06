from abc import ABC, abstractmethod, abstractmethod

from audio_data_poisoning.common.utils import DEVICE, SingletonMeta


class BaseModel(ABC, metaclass=SingletonMeta):

    def __init__(self, model, model_type: str, **kwargs):
        self.model = model.from_pretrained(
            pretrained_model_name_or_path=model_type, **kwargs
        ).to(DEVICE)

    @abstractmethod
    def get_features(self, *args, **kwargs):
        pass
