"""
End-to-end smoke tests for demo_app.py using Streamlit's AppTest harness.

demo_app.py is a script-style Streamlit app (top-level st.* calls run on
import), so it can't be unit-tested by importing individual functions the
normal way — AppTest runs the whole script under a real ScriptRunContext,
which is the supported way to test a Streamlit app end-to-end. This is what
would have caught the broken hot-linked image and the hardcoded dev-machine
path before they reached the live deployment.
"""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "demo_app.py")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    return at


def test_results_dashboard_loads_without_exception(app):
    assert not app.exception


def test_time_series_scores_page_loads(app):
    app.sidebar.radio[0].set_value("📈 Time-Series Scores").run()
    assert not app.exception


def test_live_playground_page_loads(app):
    app.sidebar.radio[0].set_value("🚨 Live Anomaly Playground").run()
    assert not app.exception


def test_no_broken_image_references(app):
    # Regression test: the sidebar used to hot-link a Wikimedia image that
    # 404'd in production. The sidebar should render with plain text/markdown
    # only now, no st.image calls.
    assert len(app.sidebar.get("image")) == 0
