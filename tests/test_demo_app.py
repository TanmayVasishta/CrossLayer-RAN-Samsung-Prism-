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


# ── Live Anomaly Playground: exhaustive farm × modality sweep ─────────────────
#
# Regression coverage for a real bug found by hand-testing: thresholds.json
# only ever held Memory+SLURM thresholds, but the playground displayed that
# same value labeled as the threshold for whichever modality was selected.
# For farm14/Disk this meant a real reconstruction error of 3.49 (well under
# the true trained Disk threshold of ~9.02) was compared against the wrong
# value (0.128, the Memory+SLURM number) and reported as a false "27x over
# threshold" anomaly. CPU/Disk each use a ReconAE (models/autoencoder.py)
# that carries its own trained .thr/.mu — the fix reads those directly
# instead of reusing thresholds.json for every modality.

FARMS = ["farm14", "farm16", "farm18", "farm19", "farm23"]
MODALITIES = ["🧠 Memory + SLURM", "⚡ CPU", "💾 Disk"]


@pytest.mark.parametrize("farm", FARMS)
@pytest.mark.parametrize("modality", MODALITIES)
def test_playground_runs_inference_for_every_farm_and_modality(modality, farm):
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("🚨 Live Anomaly Playground").run()
    at.selectbox[1].set_value(modality).run()
    at.selectbox[0].set_value(farm).run()
    at.slider[2].set_value(4.0).run()  # I/O Thrashing — reference calibration point
    at.button[0].click().run()

    assert not at.exception, f"{modality}/{farm} raised: {at.exception}"


@pytest.mark.parametrize("farm", FARMS)
@pytest.mark.parametrize("modality", MODALITIES)
def test_playground_fault_injection_is_calibrated_to_be_believable(modality, farm):
    """
    Regression test for a real UX bug: after the threshold fix above started
    reading each model's own real (and sometimes much higher) threshold, some
    farm/modality combinations stopped ever flagging an anomaly even with
    sliders maxed out — a dead, unconvincing demo. Fault injection strength is
    now calibrated per model at inference time so a slider pushed to 4 (out
    of its 0-15 range) reliably crosses that model's own real threshold,
    while 0 always stays NORMAL. Verify both ends of that contract hold for
    every farm/modality combination, not just the one that was hand-tested.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("🚨 Live Anomaly Playground").run()
    at.selectbox[1].set_value(modality).run()
    at.selectbox[0].set_value(farm).run()

    at.slider[2].set_value(0.0).run()
    at.button[0].click().run()
    assert not at.exception
    assert any("NORMAL" in e.value for e in at.success), (
        f"{modality}/{farm}: baseline (slider=0) should always read NORMAL"
    )

    at.slider[2].set_value(4.0).run()
    at.button[0].click().run()
    assert not at.exception
    assert any("ANOMALY DETECTED" in e.value for e in at.error), (
        f"{modality}/{farm}: slider=4 should reliably trigger a detected anomaly"
    )

    banner_text = " ".join(e.value for e in list(at.success) + list(at.warning))
    assert "threshold" in banner_text.lower(), (
        f"{modality}/{farm}: no threshold info shown in playground banner"
    )


def test_cpu_and_disk_thresholds_differ_from_memory_slurm():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("🚨 Live Anomaly Playground").run()
    at.selectbox[0].set_value("farm14").run()

    banners = {}
    for modality in MODALITIES:
        at.selectbox[1].set_value(modality).run()
        banner = next(
            (e.value for e in list(at.success) if "trained threshold" in e.value),
            None,
        )
        assert banner, f"no trained-threshold banner shown for {modality}"
        banners[modality] = banner

    assert banners["🧠 Memory + SLURM"] != banners["⚡ CPU"]
    assert banners["🧠 Memory + SLURM"] != banners["💾 Disk"]
    assert banners["⚡ CPU"] != banners["💾 Disk"]


def test_no_stray_warnings_during_inference(recwarn):
    # Regression test: raw numpy arrays used to be passed straight into a
    # scaler fitted on named columns, triggering a sklearn UserWarning
    # ("X does not have valid feature names") on every single inference run.
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("🚨 Live Anomaly Playground").run()
    at.selectbox[1].set_value("🧠 Memory + SLURM").run()
    at.selectbox[0].set_value("farm14").run()
    at.slider[0].set_value(5.0).run()
    at.button[0].click().run()

    assert not at.exception
    feature_name_warnings = [
        w for w in recwarn.list if "does not have valid feature names" in str(w.message)
    ]
    assert not feature_name_warnings
