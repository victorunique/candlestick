import pytest
from streamlit.testing.v1.app_test import AppTest
from unittest.mock import patch

@patch('src.app.get_redis')
@patch('requests.get')
@patch('requests.post')
def test_app_loads(mock_post, mock_get, mock_redis):
    mock_redis.return_value = None
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"prediction": "UP", "confidence": 0.9}
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"status": "success"}

    at = AppTest.from_file("src/app.py")
    try:
        at.run(timeout=10)
    except AssertionError as e:
        pass
    assert not at.exception
