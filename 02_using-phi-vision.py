from langchain_core.prompts.chat import ChatMessagePromptTemplate

def create_prompt(image_path: str) -> List[BaseMessage]:
    user_prompt = '<|user|>\n'
    assistant_prompt = '<|assistant|>\n'
    prompt_suffix = "<|end |>\n"
    prompt = f"{user_prompt}<|image_1|>\nDescribe this image.{prompt_suffix}{assistant_prompt}"
    return [HumanMessage(content=prompt, metadata={"image_path": image_path})]

# Create the prompt
image_path = "path_to_your_image.jpg"
messages = create_prompt(image_path)

# Load the model
model_name = "Phi35VisionInstructModel"
model = mlflow.pyfunc.load_model(f"models:/{model_name}/1")

# Generate the response
response = model.predict([messages])
print(response)