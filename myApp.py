from my_code.inference_from_user_prompt import inference_from_input
import gradio as gr

def generate_text(user_input):
    return inference_from_input(user_input)

demo = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(label="Enter a prompt"),
    outputs=gr.Textbox(label="Generated text"),
    title="Language Model Demo",
)

if __name__ == "__main__":
    demo.launch(share=True)

