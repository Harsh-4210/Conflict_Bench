"""Invalid config must fail with ConfigError before any expensive work."""

import pytest

from training.config import ConfigError, config_from_dict, load_config


@pytest.mark.parametrize("field", ["learning_rate", "beta", "weight_decay", "temperature", "epochs"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_numeric_values_are_rejected(field, value):
    with pytest.raises(ConfigError, match=field):
        config_from_dict({field: value})


@pytest.mark.parametrize("data", [[], [1], "model", 3, {1: "bad"}, {"model": "valid", 2: "bad"}])
def test_nonmapping_or_nonstring_config_keys_are_rejected(data):
    with pytest.raises(ConfigError):
        config_from_dict(data)


@pytest.mark.parametrize("mix", [{True: 1.0}, {1.5: 1.0}, {1: 1.0, "1": 2.0},
                                 {1: float("inf")}, {1: 1e308, 2: 1e308}])
def test_ambiguous_difficulty_keys_and_nonfinite_weights_are_rejected(mix):
    with pytest.raises(ConfigError, match="difficulty_mix"):
        config_from_dict({"train_difficulty_mix": mix})


@pytest.mark.parametrize("contents", ["model: [unterminated", "[]", "false", "0", "1: value",
                                      "learning_rate: 1e-6\nlearning_rate: 2e-6",
                                      "train_difficulty_mix: {1: 0.5, true: 0.5}",
                                      "seed: 2026-13-01", "null", ""])
def test_yaml_errors_and_nonmapping_values_are_config_errors(tmp_path, contents):
    path = tmp_path / "invalid.yaml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


@pytest.mark.parametrize("override", ["learning_rate=[bad", "seed=2026-13-01",
                                     "train_difficulty_mix={1: 0.5, true: 0.5}",
                                     "train_difficulty_mix={1: 0.5, 1: 0.5}"])
def test_malformed_override_is_a_cli_error_without_a_traceback(capsys, override):
    from training.train_grpo import main

    with pytest.raises(SystemExit) as exc:
        main(["--config", "configs/grpo_smoke.yaml", "--check-config", "--set", override])
    assert exc.value.code == 2
    assert "invalid YAML" in capsys.readouterr().err
