# Databricks notebook source
from typing import Any, Dict, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatResult
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image
import mlflow
from mlflow.models.signature import infer_signature

# COMMAND ----------

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

    def _llm_type(self) -> str:
        return "custom"

# COMMAND ----------

# Define the input and output schema
input_example = {
    "image_path": "path_to_your_image.jpg",  # This can be a path to an actual image file for testing
    "prompt": "Describe this image."
}
output_example = "A beautiful sunset over the mountains."

# Infer the model signature
signature = infer_signature(input_example, output_example)

# COMMAND ----------

# Log the model
with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        artifact_path="vision_model",
        python_model=CustomLLMChat(),
        signature=signature,
        input_example=input_example
    )

# COMMAND ----------

    # Register the model in the MLflow Model Registry
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_info.artifact_path}"
    model_name = "PhiVisionModel"
    catalog = "hive_metastore"
    schema = "labuser8941794_1736762028"

    mlflow.register_model(
        model_uri=model_uri,
        name=f"{catalog}.{schema}.{model_name}"
    )