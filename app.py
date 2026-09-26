"""Hugging Face Spaces entry point for the lunar image registration demo."""

import gradio as gr

from ui.gradio_app import demo

# The registration pipeline is CPU-only. ZeroGPU nevertheless expects a GPU-decorated
# function to be present at startup. This probe is never called and does not request a GPU.
try:
    import spaces

    @spaces.GPU(duration=1)
    def _zerogpu_startup_probe():
        return None
except ImportError:
    pass


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(theme=gr.themes.Soft())