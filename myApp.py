import gradio as gr
from my_code.inference_user_prompt import generate_from_user_input


# Call to my inference code
def gradio_generate(prompt, max_new_tokens=50, temperature=1.0):
    return generate_from_user_input(
        prompt=prompt,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
    )


# Gradio UI
demo = gr.Interface(
    fn=gradio_generate,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your prompt here..."),
        gr.Slider(10, 200, value=50, label="Max new tokens"),
        gr.Slider(0.5, 1.5, value=1.0, step=0.1, label="Temperature"),
    ],
    outputs="text",
    title="TinyStories Transformer",
    description="Enter a prompt and generate continuation with the trained TinyStories model."
)

if __name__ == "__main__":
    demo.launch()
