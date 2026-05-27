from transformers import pipeline
import gradio as gr

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_article(article, max_length, min_length):
    if len(article.strip()) == 0:
        return "Please enter an article to summarize."
    
    summary = summarizer(
        article,
        max_length=int(max_length),
        min_length=int(min_length),
        do_sample=False
    )
    return summary[0]["summary_text"]

with gr.Blocks(title="Article Summarizer") as demo:
    gr.Markdown("## Article Summarizer")
    gr.Markdown("Paste a long article below and get a concise summary.")
    
    with gr.Row():
        with gr.Column():
            article_input = gr.Textbox(
                label="Article Text",
                placeholder="Paste your article here...",
                lines=12
            )
            with gr.Row():
                max_len = gr.Slider(
                    minimum=50, maximum=500,
                    value=150, step=10,
                    label="Max Length"
                )
                min_len = gr.Slider(
                    minimum=20, maximum=200,
                    value=50, step=10,
                    label="Min Length"
                )
            summarize_btn = gr.Button("Summarize", variant="primary")
        
        with gr.Column():
            summary_output = gr.Textbox(
                label="Summary",
                lines=12
            )
    
    summarize_btn.click(
        fn=summarize_article,
        inputs=[article_input, max_len, min_len],
        outputs=summary_output
    )

if __name__ == "__main__":
    demo.launch()