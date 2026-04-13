import logging
import yaml
from pathlib import Path

from pydantic import AliasChoices, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from resume_rag.domain.models import ApplicationSettings

logger = logging.getLogger(__name__)

class EnvironmentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str = Field(
        ...,
        validation_alias=AliasChoices("API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"),
    )
    endpoint_url: str = Field(
        ...,
        validation_alias=AliasChoices("ENDPOINT_URL", "AZURE_OPENAI_ENDPOINT"),
    )
    vector_db_dir: str = "./vector-db"
    results_dir: str = "./results"
    data_dir: str = "./dataset"

class ConfigManager:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        self.config_path = Path(config_path).resolve()
        self._project_root = self.config_path.parent.parent
        env_path = self._project_root / ".env"
        env_kw = {}
        if env_path.is_file():
            env_kw["_env_file"] = str(env_path)
            logger.debug("Loading environment from %s", env_path)
        else:
            logger.warning(
                "No .env file at %s — using only OS environment variables. "
                "Create .env here or set API_KEY / ENDPOINT_URL in the environment.",
                env_path,
            )
        self.env_settings = EnvironmentSettings(**env_kw)
        self.app_settings = self._load_app_config()
        self._ensure_directories()

    @property
    def project_root(self) -> Path:
        return self._project_root

    def _load_app_config(self) -> ApplicationSettings:
        try:
            with open(self.config_path, "r") as f:
                config_data = yaml.safe_load(f)
            return ApplicationSettings(**config_data)
        except FileNotFoundError:
            logger.error("Configuration file not found: %s", self.config_path)
            raise
        except ValidationError as e:
            logger.error("Configuration validation error: %s", e)
            raise

    def _resolve_env_path(self, raw: str) -> Path:
        p = Path(raw).expanduser()
        if p.is_absolute():
            return p.resolve()
        return (self._project_root / p).resolve()

    def _ensure_directories(self) -> None:
        directories = [
            self.vector_db_dir / self.app_settings.storage.chroma_db_path,
            self.results_dir / self.app_settings.storage.evaluation_results_path,
            self.results_dir / self.app_settings.storage.logs_path,
            self.data_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def api_key(self) -> str:
        return self.env_settings.api_key

    @property
    def endpoint_url(self) -> str:
        return self.env_settings.endpoint_url

    @property
    def vector_db_dir(self) -> Path:
        return self._resolve_env_path(self.env_settings.vector_db_dir)

    @property
    def results_dir(self) -> Path:
        return self._resolve_env_path(self.env_settings.results_dir)

    @property
    def data_dir(self) -> Path:
        return self._resolve_env_path(self.env_settings.data_dir)

    @property
    def chroma_persist_dir(self) -> str:
        return str(self.vector_db_dir / self.app_settings.storage.chroma_db_path)

    @property
    def evaluation_results_dir(self) -> str:
        return str(self.results_dir / self.app_settings.storage.evaluation_results_path)

    @property
    def logs_dir(self) -> str:
        return str(self.results_dir / self.app_settings.storage.logs_path)

    def get_llm_config(self):
        return self.app_settings.llm

    def get_embedding_config(self):
        return self.app_settings.embeddings

    def get_text_splitter_config(self):
        return self.app_settings.text_splitter

    def get_access_control_config(self):
        return self.app_settings.access_control

    def get_evaluation_config(self):
        return self.app_settings.evaluation