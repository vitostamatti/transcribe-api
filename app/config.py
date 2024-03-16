from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    hf_token: str
    model_config = SettingsConfigDict(env_file=".env")
