"""Close every SQLite connection opened through the test helpers."""

import pytest

from . import helpers


@pytest.fixture(autouse=True)
def close_helper_connections():
    yield
    for connection in helpers.OPEN_CONNECTIONS:
        connection.close()
    helpers.OPEN_CONNECTIONS.clear()
