import gc
import os
import sys

import streamlit as st
import streamlit.web.cli as stcli
import torch
from streamlit import runtime

PAGES = {
    "🎛️ Home": "tasks.home",
    "🌊 Text to Audio": "tasks.text_to_audio",
    "✨ Audio to Audio": "tasks.audio_to_audio",
    "🎭 Interpolation": "tasks.interpolation",
    "✂️ Audio Splitter": "tasks.split_audio",
    "📜 Text to Audio Batch": "tasks.text_to_audio_batch",
    "📎 Sample Clips": "tasks.sample_clips",
    "␧ Spectrogram to Audio": "tasks.image_to_audio",
}


def stop_everything_and_free_gpu() -> None:
    """Clear caches, free GPU/XPU memory, and exit the server process."""
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass

    gc.collect()

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            torch.xpu.empty_cache()
            torch.xpu.synchronize()
        except Exception:
            pass

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass

    gc.collect()
    # Exit so the process fully releases the GPU
    os._exit(0)


def render() -> None:
    st.set_page_config(
        page_title="Riffusion Playground",
        page_icon="🎸",
        layout="wide",
    )

    # Top bar: stop releases GPU fully (browser close alone does not)
    _top_left, top_right = st.columns([4, 1])
    with top_right:
        if st.button(
            "⏹ Stop & Free GPU",
            type="primary",
            help="Stops the server, clears models, and fully frees your GPU/XPU.",
            use_container_width=True,
        ):
            st.warning("Stopping… freeing GPU…")
            stop_everything_and_free_gpu()

    with st.sidebar:
        st.markdown("### Session")
        if st.button(
            "⏹ Stop & Free GPU",
            key="sidebar_stop_free_gpu",
            help="Stops the server and fully frees your GPU/XPU.",
            use_container_width=True,
        ):
            st.warning("Stopping… freeing GPU…")
            stop_everything_and_free_gpu()

    page = st.sidebar.selectbox("Page", list(PAGES.keys()))
    assert page is not None
    module = __import__(PAGES[page], fromlist=["render"])
    module.render()


if __name__ == "__main__":
    if runtime.exists():
        render()
    else:
        sys.argv = ["streamlit", "run"] + sys.argv
        sys.exit(stcli.main())
