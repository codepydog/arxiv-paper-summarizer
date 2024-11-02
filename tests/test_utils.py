import pytest

from arxiv_paper_summarizer.utils import get_env_var


def test_get_env_var(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", "/home/runner")
    monkeypatch.setenv("USER", "runner")
    monkeypatch.setenv("SHELL", "/bin/bash")

    assert get_env_var("HOME") == "/home/runner"
    assert get_env_var("USER") == "runner"
    assert get_env_var("SHELL") == "/bin/bash"

    with pytest.raises(ValueError):
        get_env_var("NOT_EXISTING")
