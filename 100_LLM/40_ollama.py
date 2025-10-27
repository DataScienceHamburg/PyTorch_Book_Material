#%% packages
import base64
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

#%% configuration
MODEL_NAME = "gemma3:4b"
USER_PROMPT = "Was zeigt dieses Bild? Antworte in einem Absatz und auf Deutsch."
IMAGE_PATH = "TrainingProcess.png"

#%% helper function
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image(IMAGE_PATH)

#%% model init
model = ChatOllama(model=MODEL_NAME, temperature=0.2)

#%% multimodal invoke
message = HumanMessage(
    content=[
        {"type": "text", "text": USER_PROMPT},
        {"type": "image_url", "image_url": f"data:image/png;base64,{base64_image}"},
    ]
)

res = model.invoke([message])

# %%
res.content