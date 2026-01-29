import os
import time
import gradio as gr
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("API_MODEL", "zai-org/GLM-4.7-Flash")

client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

SYSTEM_PROMPT = (
    "You are a Lecture Notes Summarizer for WPI students. "
    "Summarize clearly using headings and bullet points. "
    "Extract key terms, definitions, and 5 practice questions at the end."
)

def summarize_notes(notes, summary_style, max_tokens, temperature):
    notes = (notes or "").strip()
    if not notes:
        return "Please paste your notes.", ""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Style: {summary_style}\n\nNotes:\n{notes}"},
    ]

    t0 = time.time()
    try:
        resp = client.chat_completion(
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        dt = time.time() - t0
        out = resp.choices[0].message["content"].strip()
        meta = f"Mode: API | Model: {MODEL_ID} | Time: {dt:.2f}s"
        return out, meta
    except Exception as e:
        return f"Error calling API: {e}", "API error"

with gr.Blocks() as demo:
    gr.Markdown("# 📝 Lecture Notes Summarizer — API Version")
    gr.Markdown("This version calls a hosted model via Hugging Face.")

    notes = gr.Textbox(label="Paste your lecture notes", lines=12, placeholder="Paste text here...")
    summary_style = gr.Dropdown(
        ["Short (bullet points)", "Detailed (headings + bullets)", "Exam-focused (key ideas + likely questions)"],
        value="Detailed (headings + bullets)",
        label="Summary style",
    )

    with gr.Row():
        max_tokens = gr.Slider(64, 800, value=350, step=16, label="Max tokens")
        temperature = gr.Slider(0.1, 1.2, value=0.4, step=0.1, label="Temperature")

    btn = gr.Button("Summarize")
    out = gr.Markdown(label="Summary")
    meta = gr.Textbox(label="Run info", interactive=False)

    btn.click(summarize_notes, [notes, summary_style, max_tokens, temperature], [out, meta])

demo.launch()
