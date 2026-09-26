"""Hugging Face Spaces entry point for the lunar image registration demo."""

import os

import gradio as gr

from ui.gradio_app import demo

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        theme=gr.themes.Soft(),
    )