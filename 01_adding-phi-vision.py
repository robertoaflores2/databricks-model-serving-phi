from typing import Any, Dict, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatResult
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image
import mlflow
from mlflow.models.signature import infer_signature

class CustomLLMChat(BaseChatModel):
    model_name: str = "microsoft/Phi-3.5-vision-instruct"
    
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_name)
    
    def _generate_response(self, image_path: str, prompt: str) -> str:
        # Load the image
        image = Image.open(image_path).convert("RGB")
        
        # Preprocess the image
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        
        # Post-process the output
        result = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return result[0]
    
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> ChatResult:
        # Extract the image path and prompt from the messages
        image_path = messages[0].metadata.get("image_path")
        prompt = messages[0].content
        
        # Generate the response
        response = self._generate_response(image_path, prompt)
        
        return ChatResult(generations=[AIMessage(content=response)])

# Define the input and output schema
input_example = {
    "image_path": "path_to_your_image.jpg",
    "prompt": "Describe this image."
}
output_example = "A beautiful sunset over the mountains."

# Infer the model signature
signature = infer_signature(input_example, output_example)

# Log the model
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="vision_model",
        python_model=CustomLLMChat(),
        signature=signature,
        input_example=input_example
    )

# Register the model in the Unity Catalog
model_uri = "runs:/<run_id>/vision_model"
model_name = "Phi35VisionInstructModel"
mlflow.register_model(model_uri, model_name)