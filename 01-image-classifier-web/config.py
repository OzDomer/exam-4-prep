from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    port: int = 9000
    app_name: str = "image-classifier"
    # Path to the checkpoint produced by train.py. Override with an env var,
    # e.g.  MODEL_PATH=./model.pt
    model_path: str = "model.pt"


def get_settings() -> Settings:
    return Settings()
