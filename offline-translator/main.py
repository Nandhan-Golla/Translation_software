from src.interface.gradio_app import create_gradio_interface

if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch()
