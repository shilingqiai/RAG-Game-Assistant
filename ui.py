import gradio as gr
from game_core import GameCompanion

companion = GameCompanion()


def chat_with_companion(message, history):
    response = companion.chat(message)
    return response


with gr.Blocks(title="Elden Ring Companion") as demo:
    gr.Markdown("# Elden Ring Game Companion")
    gr.Markdown("Ask me anything about Elden Ring - lore, bosses, combat tips, and more!")
    
    chatbot = gr.ChatInterface(
        fn=chat_with_companion,
        title="Elden Ring Companion",
        examples=[
            "Who is Malenia?",
            "How do I beat Radahn?",
            "What are the different endings?",
            "Which class should I choose?",
        ],
    )


if __name__ == "__main__":
    demo.launch()