import os
from PIL import Image
import gradio as gr

# Import existing backend service
from app.service import FaceService

print("Initializing Indian Cricket Face Identification Service...")
service = FaceService()

def predict_cricketer(input_image):
    if input_image is None:
        return "Please upload an image.", None, None

    try:
        # Run face identification
        pil_img = Image.fromarray(input_image)
        result = service.identify(pil_img, top_k=5)

        if result.get("unknown"):
            top_cand = result["top_k"][0]["identity"] if result.get("top_k") else "None"
            msg = f"❌ Rejected as Unknown.\nClosest candidate was: {top_cand} (Confidence below {result.get('threshold', 0.1)*100:.0f}% threshold)"
            return msg, None, None

        player_name = result["identity"]
        confidence = result["confidence"] * 100
        
        # Format top 5 predictions for bar chart
        candidates = {c["identity"]: float(c["confidence"]) for c in result.get("top_k", [])}
        
        summary = f"🏏 **Identified Player**: {player_name}\n🎯 **Confidence**: {confidence:.1f}%"
        return summary, candidates, f"Player: {player_name}"

    except Exception as e:
        return f"Error processing image: {str(e)}", None, None

# Gradio Interface
custom_css = """
#header { text-align: center; margin-bottom: 20px; }
"""

with gr.Blocks(title="Indian Cricket Player Face ID", css=custom_css) as demo:
    gr.Markdown("# 🏏 India XI — Cricket Player Identification AI", elem_id="header")
    gr.Markdown("Upload an image of an Indian cricketer to identify them using ArcFace embeddings & Calibrated SVM.")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="numpy", label="Upload Player Photo")
            submit_btn = gr.Button("Identify Player", variant="primary")
            
        with gr.Column():
            result_text = gr.Markdown(label="Result")
            confidence_chart = gr.Label(num_top_classes=5, label="Top Predictions & Confidence")

    submit_btn.click(
        fn=predict_cricketer,
        inputs=[img_input],
        outputs=[result_text, confidence_chart]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

