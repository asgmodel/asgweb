from django.shortcuts import render

from django.http.response import JsonResponse
from rest_framework.parsers import JSONParser
from rest_framework import status

from tutorials.models import Tutorial,Scenario
from tutorials.serializers import TutorialSerializer,MST,ASGAI,ScenarioSerializer,AIVITSSerializer, AI,create_chat_session
from rest_framework.decorators import api_view
import requests
import uuid

def remove_extra_spaces(text):
  """
  Removes extra spaces between words in a string.
  Args:
    text: The string to process.
  Returns:
    The string with extra spaces removed.
  """
  return ' '.join(text.split())



def   get_answer_ai(text):
      global AI
      try:
          response = AI.send_message(text)
          return response.text


      except :
          AI=create_chat_session()
          response = AI.send_message(text)
          return response.text
def toclean(txt):
    try:
        url="https://ansaltwyl256.pythonanywhere.com/api/nlp/"+txt
        response = requests.get(url)

        return response.json()['description']
    except:
        return txt


@api_view(['GET', 'POST', 'DELETE'])
def input(request,pk):
    print(pk)
    if request.method == 'GET':
        doc='A:'

        langs=''#ASGAI.Soft.detector.detect(pk)
        text_output,dis=MST.predictAPI(MST.to_tran(pk))

        dis=dis#MST.to_tran(dis,dest=langs.lang)


        tutorials = {"title":str(text_output),"description":dis,"published":True if str(langs)=='ar' else False}

        tutorial_serializer = TutorialSerializer(data=tutorials)

        if tutorial_serializer.is_valid():

            # tutorial_serializer.create()
            return JsonResponse(tutorial_serializer.data, status=status.HTTP_200_OK)
        return JsonResponse(tutorial_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    else:
        tutorials = {"title":str(text_output),"description":text_output,"published":True}

        tutorial_serializer = TutorialSerializer(data=tutorials)
        if tutorial_serializer.is_valid():
            tutorial_serializer.save()
            return JsonResponse(tutorial_serializer.data, status=status.HTTP_201_CREATED)
        return JsonResponse(tutorial_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET', 'POST', 'DELETE'])
def input_soft(request,pk):
    print(pk)
    if request.method == 'GET':
        doc='A:'

        langs=''#ASGAI.Soft.detector.detect(pk)
        text_output,dis=ASGAI.Soft.predictAPI(ASGAI.Soft.to_tran(pk))

        dis=dis#ASGAI.Soft.to_tran(dis,dest=langs.lang)


        tutorials = {"title":str(text_output),"description":dis +str(request.data),"published":True if str(langs)=='ar' else False}

        tutorial_serializer = TutorialSerializer(data=tutorials)

        if tutorial_serializer.is_valid():

            # tutorial_serializer.create()
            return JsonResponse(tutorial_serializer.data, status=status.HTTP_200_OK)
        return JsonResponse(tutorial_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    else:
       pass

@api_view(['GET', 'POST', 'DELETE'])
def input_group(request,pk):
    print(pk)
    if request.method == 'GET':
        doc='A:'

        langs=''#ASGAI.Group.detector.detect(pk)
        text_output,dis=ASGAI.Group.predictAPI(ASGAI.Group.to_tran(pk))

        dis=dis#ASGAI.Group.to_tran(dis,dest=langs.lang)


        tutorials = {"title":str(text_output),"description":dis,"published":True if str(langs)=='ar' else False}

        tutorial_serializer = TutorialSerializer(data=tutorials)

        if tutorial_serializer.is_valid():

            # tutorial_serializer.create()
            return JsonResponse(tutorial_serializer.data, status=status.HTTP_200_OK)
        return JsonResponse(tutorial_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    else:
       pass

@api_view(['GET', 'POST', 'DELETE'])
def search(request,pk):

    print(pk)
    if request.method == 'GET':
        doc='A:'
        Scenario.objects.all().delete()
        c=0
        userid=str(uuid.uuid4())
        def addsenario(inputs):


            data,pd=inputs
            seqtactic=''
            seqtec=''
            for ob in data[0]:
                seqtactic+=ob[0]+"$@"
                seqtec+=ob[1]+"$@"

            out,_=ASGAI.Soft.predictAPI(seqtec)
            score=out+"  is state &socre : "+data[1][1]+"   "+str(round(data[2],2))

            tutorial = {"seqtactic":seqtactic,'iduser':userid,"seqtec":seqtec,"score":score}

            serrr = ScenarioSerializer(data=tutorial)
            if serrr.is_valid():
                serrr.save()



        ASGAI.cks.onsequens=addsenario

        text_output=ASGAI.search([pk],istrans=True,numstop=20)
        allob=Scenario.objects.all()



        tutorial_serializer = ScenarioSerializer(data=allob,many=True)
        tutorial_serializer.is_valid()



        return JsonResponse(tutorial_serializer.data, safe=False)
    else:
        tutorials = {"seqtactic":'tttttttt','iduser':'user1',"seqtec":'yhtyhtyhty',"score":'0.9'}

        tutorial_serializer = ScenarioSerializer(data=tutorials)
        if tutorial_serializer.is_valid():
            tutorial_serializer.save()
            return JsonResponse(tutorial_serializer.data, status=status.HTTP_201_CREATED)
        return JsonResponse(tutorial_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET', 'POST', 'DELETE'])
def generatingai(request,pk):

    print(pk)
    if request.method == 'GET':
        doc='A:'
        Scenario.objects.all().delete()
        c=0
        userid=str(uuid.uuid4())
        def addsenario(inputs):


            data,pd=inputs
            seqtactic=''
            seqtec=''
            for ob in data[0]:
                seqtactic+=ob[0]+"$@"
                seqtec+=ob[1]+"$@"

            out,_=ASGAI.Soft.predictAPI(seqtec)
            out2,_=ASGAI.Group.predictAPI(seqtec+" "+out)
            score=out2+"@"+out+"@"+data[1][1]+"@"+str(round(data[2],2))

            tutorial = {"seqtactic":seqtactic,'iduser':userid,"seqtec":seqtec,"score":score}

            serrr = ScenarioSerializer(data=tutorial)
            if serrr.is_valid():
                serrr.save()



        ASGAI.cks.onsequens=addsenario

        text_output=ASGAI.search([pk],istrans=True,numstop=500)
        allob=Scenario.objects.filter(iduser=userid)

        tutorials = {"title":str(len(allob)),"description":userid,"published":True }

        tutorial_serializer = TutorialSerializer(data=tutorials)

        if tutorial_serializer.is_valid():

            # tutorial_serializer.create()
            return JsonResponse(tutorial_serializer.data, status=status.HTTP_200_OK)
        return JsonResponse(tutorial_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    else:
        tutorials = {"seqtactic":'tttttttt','iduser':'user1',"seqtec":'yhtyhtyhty',"score":'0.9'}

        tutorial_serializer = ScenarioSerializer(data=tutorials)
        if tutorial_serializer.is_valid():
            tutorial_serializer.save()
            return JsonResponse(tutorial_serializer.data, status=status.HTTP_201_CREATED)
        return JsonResponse(tutorial_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET', 'POST', 'DELETE'])
def searchall(request,pk):
    print(pk)
    if request.method == 'GET':
        doc='A:'
      #  Scenario.objects.all().delete()
        c=0
        def addsenario(inputs):


            data,pd=inputs
            seqtactic=''
            seqtec=''
            for ob in data[0]:
                seqtactic+=ob[0]+"$@"
                seqtec+=ob[1]+"$@"

            out,_=ASGAI.Soft.predictAPI(seqtec)
            score=out+"  is state &socre : "+data[1][1]+"   "+str(round(data[2],2))

            tutorial = {"seqtactic":seqtactic,'iduser':'user'+str(10),"seqtec":seqtec,"score":score}

            serrr = ScenarioSerializer(data=tutorial)
            if serrr.is_valid():
                serrr.save()



        ASGAI.cks.onsequens=addsenario

        text_output=ASGAI.search([pk],istrans=True)
        allob=Scenario.objects.all()



        tutorial_serializer = ScenarioSerializer(data=allob,many=True)
        tutorial_serializer.is_valid()



        return JsonResponse(tutorial_serializer.data, safe=False)
    else:
        tutorials = {"seqtactic":'tttttttt','iduser':'user1',"seqtec":'yhtyhtyhty',"score":'0.9'}

        tutorial_serializer = ScenarioSerializer(data=tutorials)
        if tutorial_serializer.is_valid():
            tutorial_serializer.save()
            return JsonResponse(tutorial_serializer.data, status=status.HTTP_201_CREATED)
        return JsonResponse(tutorial_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET', 'POST', 'DELETE'])
def inputpipline(request,pk):
    if request.method == 'GET':
        doc='A:'

        langs='' #ASGAI.Soft.detector.detect(pk)
        text_output,_,dis=ASGAI.Group.Predict_ALL(ASGAI.Group.to_tran(pk))
        items=text_output.split('--')

        text_output= "Technique:"+items[1]+",Incident: "+items[2]+", Group:"+ dis['svmK']
        txtdes= items[1]+" " +ASGAI.Tec.DES[items[1]][0:500]+"  @@$ "+items[2]+" :" +ASGAI.Soft.DES[items[2]][0:500]+"   @@$  "+ASGAI.Group.DES[dis['svmK']][0:500]
        langs='en' #langs.lang
        # if langs.lang!='en':
        #   txtdes=MST.to_tran(txtdes,dest=langs.lang)



        tutorials = {"title":str(text_output),"description":txtdes,"published":True if str(langs)=='ar' else False}

        tutorial_serializer = TutorialSerializer(data=tutorials)

        if tutorial_serializer.is_valid():

            # tutorial_serializer.create()
            return JsonResponse(tutorial_serializer.data, status=status.HTTP_200_OK)
        return
        # return JsonResponse(tutorial_serializer.errors, status=status.HTTP_400_BAD_REQUEST)onResponse(tutorial_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    else:
        tutorials = {"title":str(text_output),"description":text_output,"published":True}

        tutorial_serializer = TutorialSerializer(data=tutorials)
        if tutorial_serializer.is_valid():
            tutorial_serializer.save()
            return JsonResponse(tutorial_serializer.data, status=status.HTTP_201_CREATED)
        return JsonResponse(tutorial_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET', 'POST', 'DELETE'])
def search2(request,pk):
    print(pk)
    if request.method == 'GET':
        doc='A:'

        allob=Scenario.objects.filter(iduser=pk)





        tutorial_serializer = ScenarioSerializer(data=allob,many=True)
        tutorial_serializer.is_valid()



        return JsonResponse(tutorial_serializer.data, safe=False)
    else:
        tutorials = {"seqtactic":'tttttttt','iduser':'user1',"seqtec":'yhtyhtyhty',"score":'0.9'}

        tutorial_serializer = ScenarioSerializer(data=tutorials)
        if tutorial_serializer.is_valid():
            tutorial_serializer.save()
            return JsonResponse(tutorial_serializer.data, status=status.HTTP_201_CREATED)
        return JsonResponse(tutorial_serializer.errors, status=status.HTTP_400_BAD_REQUEST)




@api_view(['GET', 'POST', 'DELETE'])
def input_info_Group(request,pk):
    print(pk)
    if request.method == 'GET':
        doc='A:'

        text_output=ASGAI.Group.getLables()




        tutorials = {"title":str(text_output[0]),"description":str(text_output),"published":True}

        tutorial_serializer = TutorialSerializer(data=tutorials)

        if tutorial_serializer.is_valid():

            # tutorial_serializer.create()
            return JsonResponse(tutorial_serializer.data, status=status.HTTP_200_OK)
        return JsonResponse(tutorial_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    else:
        tutorials = {"title":str(text_output),"description":text_output,"published":True}

        tutorial_serializer = TutorialSerializer(data=tutorials)
        if tutorial_serializer.is_valid():
            tutorial_serializer.save()
            return JsonResponse(tutorial_serializer.data, status=status.HTTP_201_CREATED)
        return JsonResponse(tutorial_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET', 'POST', 'DELETE'])
def input_info_Soft(request,pk):
    print(pk)
    if request.method == 'GET':
        doc='A:'
        lste=[]
        langs='a' #ASGAI.Soft.detector.detect(pk)

        text_output=ASGAI.Soft.getLables()




        tutorials = {"title":str(langs),"description":str(text_output),"published":True}

        tutorial_serializer = TutorialSerializer(data=tutorials)

        if tutorial_serializer.is_valid():

            # tutorial_serializer.create()
            return JsonResponse(tutorial_serializer.data, status=status.HTTP_200_OK)
        return JsonResponse(tutorial_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    else:
        tutorials = {"title":str(text_output),"description":text_output,"published":True}

        tutorial_serializer = TutorialSerializer(data=tutorials)
        if tutorial_serializer.is_valid():
            tutorial_serializer.save()
            return JsonResponse(tutorial_serializer.data, status=status.HTTP_201_CREATED)
        return JsonResponse(tutorial_serializer.errors, status=status.HTTP_400_BAD_REQUEST)




def  getsubdes(des):
    nl=len(des)-1
    return des[0:(500  if nl>500 else nl)]
#source .virtualenvs/myprojectvenv/bin/activate
@api_view(['GET', 'POST', 'DELETE'])
def transe(request,pk):
    print(pk)
    if request.method == 'GET':
        doc='A:'
        lste=[]
        langs='' #ASGAI.Soft.detector.detect(pk)
        txtp=ASGAI.Group.to_tran(pk)
        text_output,_,dis=ASGAI.Group.Predict_ALL(txtp)
        items=text_output.split('--')

        text_output= "Technique:"+items[1]+",Incident: "+items[2]+", Group:"+ dis['svmK']

        txtdes= items[1]+" " +ASGAI.Tec.DES[items[1]][0:500]+"  @@$ "+items[2]+" :" +ASGAI.Soft.DES[items[2]][0:500]+"   @@$  "+ASGAI.Group.DES[dis['svmK']][0:500]



        try:
            url = "https://chatgpt-gpt4-ai-chatbot.p.rapidapi.com/ask"

            # payload = { "query": txtp  }
            # headers = {
            # 	"content-type": "application/json",
            # 	"X-RapidAPI-Key": "e8af95d120msha76214f99ebe838p1ad208jsnda6e1e2cd7d9",
            # 	"X-RapidAPI-Host": "chatgpt-gpt4-ai-chatbot.p.rapidapi.com"
            # }

            # response = requests.post(url, json=payload, headers=headers)

            # url = "https://open-ai21.p.rapidapi.com/conversationgpt"

            # payload = { "messages": [
            # 		{
            # 			"role": "user",
            # 			"content":  txtp +"I asked the ASG model and his answer was as follows ( "+txtdes+" )  want you to arrange the explanation neatly without mentioning any side details  "
            # 		}
            # 	] }
            # headers = {
            # 	"content-type": "application/json",
            # 	"X-RapidAPI-Key": "e8af95d120msha76214f99ebe838p1ad208jsnda6e1e2cd7d9",
            # 	"X-RapidAPI-Host": "open-ai21.p.rapidapi.com"
            # }

            # response = requests.post(url, json=payload, headers=headers)
            # txtdes=response.json()['response']
        except:pass
        nl=len(txtdes)
      #  txtdes=txtdes[0:100 if nl>100 else nl]
        # if langs.lang!='en':
        #   txtdes=MST.to_tran(txtdes,dest=langs.lang)
        langs=''
        tutorials = {"title":text_output,"description":txtdes,"published":True if str(langs)=='ar' else False}

        tutorial_serializer = TutorialSerializer(data=tutorials)

        if tutorial_serializer.is_valid():

            # tutorial_serializer.create()
            return JsonResponse(tutorial_serializer.data, status=status.HTTP_200_OK)
        return JsonResponse(tutorial_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    else:
       pass



@api_view(['GET', 'POST', 'DELETE'])
def nlpto(request,pk):
    print(pk)
    if request.method == 'GET':


        tutorials = {"title":"yes","description":str(request.data),"published":True}

        tutorial_serializer = TutorialSerializer(data=tutorials)

        if tutorial_serializer.is_valid():

            # tutorial_serializer.create()
            return JsonResponse(tutorial_serializer.data, status=status.HTTP_200_OK)
        return JsonResponse(tutorial_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    else:
       pass


@api_view(['GET', 'POST', 'DELETE'])
def getteck(request,pk):
    print(pk)
    if request.method == 'GET':

        tree=ASGAI.SGT.getordertactics()


        text=''
        istec=True
        try:
            index=int(pk)
            for ob in tree[index].Techniques:
                text=text+"@@"+ob.Name
        except:
            for ob in tree:
                text +="@@"+ob.Name
                istec=False


        tutorials = {"title":str(uuid.uuid4()),"description":str(text),"published":istec}

        tutorial_serializer = TutorialSerializer(data=tutorials)

        if tutorial_serializer.is_valid():

            # tutorial_serializer.create()
            return JsonResponse(tutorial_serializer.data, status=status.HTTP_200_OK)
        return JsonResponse(tutorial_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    else:
       pass
from gradio_client import Client
from django.http import FileResponse

@api_view(['GET', 'POST', 'DELETE'])
def get_answer_ai(request,pk):
    if request.method == 'GET':

        text =pk
        client = Client("asg2024/wasm-speeker-sa")
        result = client.predict(
            text=text,
            name_model="asg2024/vits-ar-sa-huba",
            api_name="/generate_audio_ai"
        )

        # Assume 'result' is the filename of the generated audio
        try:
            return FileResponse(open(result, 'rb'), content_type='audio/mpeg')
        except FileNotFoundError:
            return Response(
                {"error": "Audio file not found"},
                status=status.HTTP_404_NOT_FOUND
            )

from transformers import AutoTokenizer,VitsModel
import torch
models= {}
tokenizer = AutoTokenizer.from_pretrained("asg2024/vits-ar-sa-huba",token='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')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def  get_model(name_model):
    global models
    if name_model in   models:
        return models[name_model]
    models[name_model]=VitsModel.from_pretrained(name_model,token='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').to(device)
    return models[name_model]

def  genrate_speech(text,name_model):

    inputs=tokenizer(text,return_tensors="pt")
    model=get_model(name_model)
    with torch.no_grad():
         wav=model(
             input_ids= inputs.input_ids.to(device),
             attention_mask=inputs.attention_mask.to(device),
             speaker_id=0
             ).waveform.cpu().numpy().reshape(-1)
    return model.config.sampling_rate,remove_noise_nr(wav)
import soundfile as sf

from django.http import StreamingHttpResponse
def file_iterator(file, chunk_size=1024 * 1024):
    with open(file, 'rb') as f:
        while True:
            time.sleep(0.1) # <-- add sleep
            c = f.read(chunk_size)
            if c:
                yield c
            else:
                break
from gradio_client import Client
@api_view(['GET', 'POST', 'DELETE'])
def get_answer_ai2(request,pk):
    if request.method == 'GET' :

       # text=get_answer_ai(str(pk))
        try:
            text=pk

            # text=remove_extra_spaces(text)
            if True:


                client = Client("wasmdashai/wasm-spad",download_files=False)

                result = client.submit(
                        text=text,
                        api_name="/text_to_speech",

                )
                for chk in result:
                    print(chk)
                    break

                return JsonResponse({"url":chk['url']},status=status.HTTP_200_OK)

              #  return FileResponse(open("test.wav", 'rb'), content_type='audio/mpeg')
        except FileNotFoundError:
                return Response(
                    {"error": "Audio file not found"},
                    status=status.HTTP_404_NOT_FOUND
                )

from rest_framework.views import APIView




class OutModelVITSView(APIView):
    serializer_class=AIVITSSerializer


        # 2. Create
    def get(self, request,*args, **kwargs):
        '''
        Create the Todo with given todo data
        '''
        serializer=self.serializer_class(data=request.data)
        try:
            if serializer.is_valid():


                # Assume 'result' is the filename of the generated audio

                data=serializer.data
                text='السلام علييكم'#data['text']
                client = Client("asg2024/wasm-speeker-sa")
                result = client.predict(
                    text=text,
                    model_choice="asg2024/vits-ar-sa",
                    api_name="/generate_audio_ai"
                )
                return FileResponse(open(result, 'rb'), content_type='audio/mpeg')
        except FileNotFoundError:
                return Response(
                    {"error": "Audio file not found"},
                    status=status.HTTP_404_NOT_FOUND
                )

from django.shortcuts import redirect


@api_view(['GET'])
def redirect_view(request):
    return redirect('http://asgmodel-002-site1.etempurl.com/')  # استبدل برابط الموقع الذي تريده

import torch
from typing import Any, Callable, Optional, Tuple, Union,Iterator
import numpy as np
import torch.nn as nn # Import the missing module



def _inference_forward_stream(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        speaker_embeddings: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        padding_mask: Optional[torch.Tensor] = None,
        chunk_size: int = 32,  # Chunk size for streaming output
    ) -> Iterator[torch.Tensor]:
        """Generates speech waveforms in a streaming fashion."""
        if attention_mask is not None:
            padding_mask = attention_mask.unsqueeze(-1).float()
        else:
            padding_mask = torch.ones_like(input_ids).unsqueeze(-1).float()



        text_encoder_output = self.text_encoder(
            input_ids=input_ids,
            padding_mask=padding_mask,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = text_encoder_output[0] if not return_dict else text_encoder_output.last_hidden_state
        hidden_states = hidden_states.transpose(1, 2)
        input_padding_mask = padding_mask.transpose(1, 2)

        prior_means = text_encoder_output[1] if not return_dict else text_encoder_output.prior_means
        prior_log_variances = text_encoder_output[2] if not return_dict else text_encoder_output.prior_log_variances

        if self.config.use_stochastic_duration_prediction:
            log_duration = self.duration_predictor(
                hidden_states,
                input_padding_mask,
                speaker_embeddings,
                reverse=True,
                noise_scale=self.noise_scale_duration,
            )
        else:
            log_duration = self.duration_predictor(hidden_states, input_padding_mask, speaker_embeddings)

        length_scale = 1.0 / self.speaking_rate
        duration = torch.ceil(torch.exp(log_duration) * input_padding_mask * length_scale)
        predicted_lengths = torch.clamp_min(torch.sum(duration, [1, 2]), 1).long()


        # Create a padding mask for the output lengths of shape (batch, 1, max_output_length)
        indices = torch.arange(predicted_lengths.max(), dtype=predicted_lengths.dtype, device=predicted_lengths.device)
        output_padding_mask = indices.unsqueeze(0) < predicted_lengths.unsqueeze(1)
        output_padding_mask = output_padding_mask.unsqueeze(1).to(input_padding_mask.dtype)

        # Reconstruct an attention tensor of shape (batch, 1, out_length, in_length)
        attn_mask = torch.unsqueeze(input_padding_mask, 2) * torch.unsqueeze(output_padding_mask, -1)
        batch_size, _, output_length, input_length = attn_mask.shape
        cum_duration = torch.cumsum(duration, -1).view(batch_size * input_length, 1)
        indices = torch.arange(output_length, dtype=duration.dtype, device=duration.device)
        valid_indices = indices.unsqueeze(0) < cum_duration
        valid_indices = valid_indices.to(attn_mask.dtype).view(batch_size, input_length, output_length)
        padded_indices = valid_indices - nn.functional.pad(valid_indices, [0, 0, 1, 0, 0, 0])[:, :-1]
        attn = padded_indices.unsqueeze(1).transpose(2, 3) * attn_mask

        # Expand prior distribution
        prior_means = torch.matmul(attn.squeeze(1), prior_means).transpose(1, 2)
        prior_log_variances = torch.matmul(attn.squeeze(1), prior_log_variances).transpose(1, 2)

        prior_latents = prior_means + torch.randn_like(prior_means) * torch.exp(prior_log_variances) * self.noise_scale
        latents = self.flow(prior_latents, output_padding_mask, speaker_embeddings, reverse=True)

        spectrogram = latents * output_padding_mask

        for i in range(0, spectrogram.size(-1), chunk_size):
            with torch.no_grad():
                wav=self.decoder(spectrogram[:,:,i : i + chunk_size] ,speaker_embeddings)
            yield wav.squeeze().cpu().numpy()


import noisereduce as nr

def remove_noise_nr(audio_data,sr=16000):
    """يزيل الضوضاء باستخدام مكتبة noisereduce."""
    reduced_noise = nr.reduce_noise(y=audio_data, sr=sr,n_jobs=8)
    return reduced_noise

def generate_audio(text, speaker_id=None):
    inputs = tokenizer(text, return_tensors="pt")#.input_ids

    speaker_embeddings = None
    #torch.cuda.empty_cache()
    model=get_model('asg2024/vits-ar-sa-huba')
    with torch.no_grad():
        for chunk in _inference_forward_stream(model,input_ids=inputs.input_ids,attention_mask=inputs.attention_mask,speaker_embeddings= speaker_embeddings,chunk_size=128):
            yield  16000,remove_noise_nr(chunk)#.astype(np.int16).tobytes()