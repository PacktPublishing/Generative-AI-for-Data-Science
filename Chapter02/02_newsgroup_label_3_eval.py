import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.generative_models as generative_models


def multiturn_generate_content():
  vertexai.init(project="299564389709", location="us-central1")
  model = GenerativeModel(
    "projects/299564389709/locations/us-central1/endpoints/63895919224946688",
  )
  return model.start_chat()


generation_config = {
    "max_output_tokens": 2048,
    "temperature": 1,
    "top_p": 1,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}
4
chat = multiturn_generate_content()


print(chat.send_message("say hi"))