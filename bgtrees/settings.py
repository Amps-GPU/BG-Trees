from dataclasses import dataclass


@dataclass
class Settings:
    use_gpu: bool = False
    D: int = 4


settings = Settings()
