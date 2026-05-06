"""Tests for the __main__ module."""

import pytest
from unittest.mock import patch


class TestMain:
    def test_main_entry(self):
        from mosqlient import cli

        with patch("sys.argv", ["mosqlient"]):
            result = cli.main([])
            assert result == 0

    def test_main_with_args(self, capsys):
        from mosqlient import cli

        with pytest.raises(SystemExit):
            cli.main(["--help"])
