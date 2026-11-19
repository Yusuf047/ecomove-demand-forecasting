from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_all_pages_and_scenario_submission():
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    assert len(app.metric) == 4
    app.radio[0].set_value("Scenario planner").run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric) == 2
    for page in ["Model evidence", "Data & methodology"]:
        app.radio[0].set_value(page).run()
        assert not app.exception


def test_empty_filter_is_handled_without_crashing():
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert "No records" in app.info[0].value
