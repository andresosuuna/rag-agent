from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    groq_api_key: str
    chroma_path: str = "./chroma_db"
    debug: bool = False

    model_config = {"env_file": ".env"}

settings = Settings()