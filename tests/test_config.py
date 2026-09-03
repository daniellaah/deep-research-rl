from pathlib import Path

import pytest

from deep_research_rl.config import ConfigError, load_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = REPOSITORY_ROOT / "configs" / "baseline.toml"


def test_baseline_contract_defaults_load() -> None:
    config = load_config(BASELINE_CONFIG)

    assert config["schema_version"] == 1
    assert config["config_kind"] == "defaults"
    assert config["dataset"] == {
        "name": "hotpot_qa",
        "variant": "distractor",
        "revision": "UNRESOLVED",
    }
    assert config["agent"]["actions"] == ["SEARCH", "ANSWER"]
    assert config["agent"]["context_policy"] == "append_only"
    assert config["agent"]["max_policy_searches"] == 5
    assert config["reward"]["intermediate"] == 0.0
    assert config["reward"]["search_cost"] == 0.0
    assert config["reward"]["token_cost"] == 0.0
    assert config["credit"]["assignment"] == "terminal_only"
    assert config["training"]["algorithm"] == "grpo"


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ('config_kind = "defaults"\n', "schema_version must be an integer"),
        ('schema_version = 1\nconfig_kind = "draft"\n', "config_kind must be one of"),
        ('schema_version = 1\nconfig_kind = ["defaults"]\n', "config_kind must be one of"),
    ],
)
def test_invalid_config_envelope(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(config_path)


def test_rejects_non_toml_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ConfigError, match=r"must be a \.toml file"):
        load_config(config_path)
