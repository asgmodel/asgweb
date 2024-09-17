from rest_framework import serializers
from tutorials.models import Tutorial,Scenario
# from django.urls import ModelPums
import sys
sys.path.insert(0, "/home/asgmodel/django-rest-api/DjangoRestApi/tutorials/ASG")


from TypeModels import TypeModel
from BasePath import BASEPATH
from ModelTEC import *
from ASGModels import ASG


MST=TEC(typemodel="svmL",model_setting=TypeModel['technique'])
MST.loadmodel()
ASGAI=ASG()
class TutorialSerializer(serializers.ModelSerializer):

    class Meta:
        model = Tutorial
        fields = ('id',
                  'title',
                  'description',
                  'published')

class ScenarioSerializer(serializers.ModelSerializer):

    class Meta:
        model = Scenario
        fields = ('id',
                  'seqtactic',
                  'iduser',
                  'seqtec','score')



class AIVITSSerializer(serializers.Serializer):
    text=serializers.CharField(max_length=2000)
    type_model= serializers.CharField(max_length=255) # Maybe add
    # class Meta:
    #     fields = ('text',
    #               'type_model')



import os

import google.generativeai as genai
api_key ='AIzaSyC85_3TKmiXtOpwybhSFThZdF1nGKlxU5c' #os.environ.get("id_gmkey")

genai.configure(api_key=api_key)

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-pro",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
)

def create_chat_session():
    chat_session = model.start_chat(
                  history=[
                    {
                      "role": "user",
                      "parts": [
                        "السلام عليكم اريد منك ان ترد على اسئلتي  دائما باللهجة السعودية النجدية  \n\n",
                      ],
                    },
                    {
                      "role": "model",
                      "parts": [
                        "هلا والله، إسأل ما في خاطرك وأنا حاضر أساعدك، بس بشرط واحد، أسئلتك تكون واضحة عشان أفهم عليك عدل وأعطيك الجواب الزين. قل وش تبي وأنا حاضر! \n",
                      ],
                    },
                    {
                      "role": "user",
                      "parts": [
                        "كيف حالك اخبارك\n",
                      ],
                    },
                    {
                      "role": "model",
                      "parts": [
                        "هلا والله وغلا، أنا طيب وبخير الحمد لله،  انت كيفك؟ عساك طيب؟ \n \n وش عندك أخبار؟ عسى كلها زينة.  \n",
                      ],
                    },
                    {
                      "role": "user",
                      "parts": [
                        "اريد ايضا ان تكون اجابتك مختصره على سبيل المثال ااكثر اجابة سطرين\n",
                      ],
                    },
                    {
                      "role": "model",
                      "parts": [
                        "خلاص، فهمتك. من عيوني، أسئلتك من اليوم وطالع أجوبتها ما تتعدى سطرين.  \n \n إسأل وشف! \n",
                      ],
                    },
                  ]
                )
    return chat_session

AI=create_chat_session()




