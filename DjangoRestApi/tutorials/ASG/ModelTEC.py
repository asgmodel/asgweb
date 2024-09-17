# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 11:27:28 2022

@author: amaer
"""

import pickle
import  numpy as np


from sklearn.feature_extraction.text import TfidfVectorizer
import regex as re
# import spacy
# from spacy import displacy
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,f1_score, precision_score, recall_score

# from googletrans import Translator

import requests
class  ModelPums:
    def __init__(self):
        self.obData=None
        self.obVec=None
        self.obSVML=None
        self.obSVMK=None
        self.oblogict=None
        self.X=None
        self.Y=None
        self.Classes=None
class TEC:
    __Outs={'svmL':None,'svmK':None,'logstick':None,'target':None}

    def createOuts():
        ob=TEC.__Outs.copy()
        for key in ob:
            ob[key]=[]
        return ob
    def __init__(self,typemodel="svmL",model_setting=None,spt=0.3,nlp=None):

        self.model_setting=model_setting
        self.isLoad=False
        self.nlp=nlp


    def toclean(self,txt):
        try:
            url="https://ansaltwyl256.pythonanywhere.com/api/nlp/"+txt
            response = requests.get(url)

            return response.json()['description']
        except:
            return "$"
    def loadmodel(self):
        mm=ModelPums()
        matrck=pickle.load(open(self.model_setting.path_model,'rb'))
        self.DES=pickle.load(open(self.model_setting.path_Qwords,'rb'))
        mm.obData=matrck['obData']
        mm.obVec=TfidfVectorizer(norm='l2')
        mm.obVec.fit(mm.obData)
        mm.obSVML=matrck['obSVML']
        mm.obSVMK=matrck['obSVMK']
        mm.oblogict=matrck['oblogict']
        mm.X=matrck['X']
        mm.Y=matrck['Y']
        mm.Classes=matrck['Classes']
        self.obMP=mm
        if self.model_setting.path_Qwords!="":
            self.Qwords={}#pickle.load(open(model_setting.path_Qwords,'rb'))
        else:
            self.Qwords={}
        self.detector =None
        # self.Splits(spt)
        self.Model=self.obMP.obSVML

        self.name=self.model_setting.name
        self.pipeline= None
        self.isLoad=True
        # self.Training()

    def getLables(self):
        return self.obMP.Classes

    def is_found(self,words,ob):
        sms=[]

        ob=ob.lower().strip()
        for w in words:
            if w==ob:
                return 1,w
            sms.append(self.similarity(w,ob))
        index=np.argmax(sms)
        if sms[index]>0.7:
            return 2, list(words)[index]

        return 0,''

    def is_found_K(self,words,ob):

        ob=ob.lower().strip()
        for w in words:
            if w==ob:
                return 1,w

        return -1,''


    def Training(self):
        self.obMP.obSVMK.fit(self.obMP.X,self.obMP.Y)


    def clean_dataT(self,data,typw='',is_input=False):
        d,_=self.clean_dataAPI(data)
        return d
        datac=data
        dock=[]
        is_found= self.is_found if is_input==True else self.is_found_K

        reg=r"([0-9])|(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"
        for datac in data:
            strr=str(datac)
            strr=re.sub(reg, "", strr)
            doc=self.nlp(strr)
            disc=""
            for token in doc:
                if not token.is_punct and not token.is_stop and not token.is_space and token.is_alpha:
                    if token.pos_=='ADJ' or  token.pos_=='NOUN' or token.pos_=='VERB':#token.pos_== typw:
                        qk,key=is_found(self.Qwords,token.lemma_)
                        if qk==1:
                            disc=disc+self.Qwords[token.lemma_]+" "
                        elif qk==2:
                            disc=disc+self.Qwords[key]+" "
                        elif qk==-1:
                            disc=disc+token.lemma_+" "

            disc=disc.lower().strip()
            if len(disc)>0:
                dock.append(disc)

        return dock

    def similarity(self,ob1,ob2):
        ob1=ob1
        ob2=ob2
        nob1=self.nlp(ob1)
        return nob1.similarity(self.nlp(ob2))

    #-----------------#
    def clean_data(self,data,typw='',is_input=False):
        return self.clean_dataAPI(data)
        datac=data
        dock=[]
        labels=[]
        is_found= self.is_found if is_input==True else self.is_found_K
        reg=r"([0-9])|(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"
        for (datac,label) in data:
            strr=str(datac)

            strr=re.sub(reg, "", strr)
            doc=self.nlp(strr)
            disc=""
            for token in doc:
                if not token.is_punct and not token.is_stop and not token.is_space and token.is_alpha:
                    if True: # token.pos_=='ADJ' or  token.pos_=='NOUN' or token.pos_=='VERB' :
#                         disc=disc+token.lemma_+" "
                        qk,key=is_found(self.Qwords,token.lemma_)
#                         print(qk,key)
                        if qk==1:
                            disc=disc+self.Qwords[key]+" "
                        elif qk==2:
                            disc=disc+self.Qwords[key]+" "
                        elif qk==-1:
                            disc=disc+token.lemma_+" "

            if len(disc)>2:
                dock.append(disc.strip())
                labels.append(label)
        return dock,labels
 #-----------------#
    def clean_dataAPI(self,data,typw='',is_input=False):
        txt=self.toclean(data)
        if txt !="$":
            return [txt],6


        reg=r"([0-9])|(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"

        data=str(data)

        strr=re.sub(reg, " ", data)
        return [strr],6
    def setPipeline(self,model=None):
        self.pipeline=model
    def to_tran(self,text,dest='en'):
        ff=True
        c=0

        while ff:


            try:
                t=self.detector.translate(str(text),dest=dest)
                ff=False
                text=t.text
            except:
                c+=1
                if c==5:
                    ff=False
                    print(' no connenet  Tr')

        return text

    def Predict_ALL(self,description,istrans=False):
        if istrans:
            description=self.to_tran(description)


        if self.pipeline is not None:
            text_output,_,outs=self.pipeline.Predict_ALL(description)
            for key in outs:
                text_output=text_output+'--'+ outs[key]
        else:
            text_output=''




        mx,outs=self.get_ptedict_proba(description+' '+text_output)

        return text_output,mx,outs


    def predictAPI(self,description,is_input=False,istrans=False):
        if istrans:
            description=self.to_tran(description)

        clean_description,_=self.clean_dataAPI(description,'',is_input=is_input)

        try:
            features=self.obMP.obVec.transform(clean_description)
            yp=self.Model.predict(features)
            txttec=self.obMP.Classes[yp[0]]
            dis=self.DES[txttec]
        except:
            txttec='No'
            dis=" ....."
        return txttec,dis
    def predict(self,description,is_input=False,istrans=False):
        if istrans:
            description=self.to_tran(description)

        clean_description,_=self.clean_data([(description,' ')],'',is_input=is_input)

        try:
            features=self.obMP.obVec.transform(clean_description)
            yp=self.Model.predict(features)
            txttec=self.obMP.Classes[yp[0]]
        except:
            txttec='No'
        return txttec

    def ptedict_proba(self,description,mx=[],is_input=False,istrans=False):
        if istrans:
            description=self.to_tran(description)

        clean_description,_=self.clean_data([(description,' ')],'',is_input=is_input)
        try:
            features=self.obMP.obVec.transform(clean_description)
            mx=self.Model.predict_proba(features)
            print(np.int16(mx*100))
            yp=np.argmax(mx)
            print(yp)

            txttec='Technique : '+ self.obMP.Classes[yp]
        except:
            txttec='No Found technique ...! (^_^)'
        return txttec



    def get_ptedict_proba(self,description,mx=[]):
        clean_description,_=self.clean_data([(description,' ')],'')

        try:
            features=self.obMP.obVec.transform(clean_description)
            mx=self.obMP.obSVML.predict_proba(features)
            yk=self.obMP.obSVMK.predict(features)


         #   yp=np.argmax(mx)


            outputs={'svmK':self.obMP.Classes[yk[0]]}
        except:
            txttec='No Found technique ...! (^_^)'
            outputs={}
        return mx,outputs

    def get_ptedict_threemodel(self,description):
        clean_description,_=self.clean_data([(description,' ')],'')

        try:
            features=self.obMP.obVec.transform(clean_description)
            yl=self.obMP.obSVML.predict(features)
            yk=self.obMP.obSVMK.predict(features)
            ym=self.obMP.oblogict.predict(features)


            outputs={'svmL':self.obMP.Classes[yl[0]],
                     'svmK':self.obMP.Classes[yk[0]],
                      'logstick':self.obMP.Classes[ym[0]]
                    }
        except:
            print('No Found technique ...! (^_^)' )
            outputs={}
        return outputs


    def verification(self,inputs=[],outputs=[]):
        out_prodect=TEC.createOuts()
        unprocess=0
        meta={"tf":TEC.createOuts(),"num":TEC.createOuts()}
        names=list(self.obMP.Classes)
        for i in range(len(outputs)):
            try:
                outs=self.get_ptedict_threemodel(inputs[i])
                target=outputs[i].strip()
                names.index(target)
                outs['target']=target


                for  key in outs:

                    out_prodect[key].append(outs[key])
                    meta['tf'][key].append(int(outs[key]==target))
                    meta['num'][key].append(names.index(outs[key]))




            except :
                unprocess+=1

        scores={}
        for key in  meta['num']:
            if key!='target':
                scores[key]=self.valmodel(meta['num']['target'],meta['num'][key],' model '+key)


        return out_prodect,meta,scores
    def valmodel(self,y,yp,titel=" "):
        print('---------------'+titel+'------------------------' )
        cr=classification_report(y, yp)
        print(cr)
        scores={}
        scores['accuracy']=accuracy_score(y, yp)
        scores['f1_score']=f1_score(y, yp, average="macro")
        scores['precision']=precision_score(y, yp, average="macro")
        scores['recall']=recall_score(y, yp, average="macro")
        return {'smmray':cr,'scores':scores}
    #----------------------
    def ChangeModel(self,ob=None):
        if ob==None:return
        if type(ob) is not str:
            self.Model=ob
        else :
            if ob=='svmL':
                self.Model=self.obMP.obSVML
            elif ob=='svmK':
                self.Model=self.obMP.obSVMK
            else:
                self.Model=self.obMP.oblogict

   #------------------------------

    def Search(self):
        txt=input('Enter any text :')
        print('Technique:'+ self.predict(txt))


    def Info_Models(self):
        print('Number Data is ',self.obMP.X.shape)
        print('Number of classes :',len(self.obMP.Classes))
        print ('---------simples -------------------')
        n=len(self.obMP.Classes)
        f=n%2
        n=n-f
        for i in range(0,n,2):
            print( (self.obMP.Classes[i],  np.sum(np.int16(self.obMP.Y==i))),'------------------',(self.obMP.Classes[i+1],  np.sum(np.int16(self.obMP.Y==i+1))))

        if f==1: print( (self.obMP.Classes[i+2],  np.sum(np.int16(self.obMP.Y==i+2))))

    def DlistModel(self):
        print ('SVC(kernel=\'linear\')-> svmL')
        print('LinearSVC(C=1.0) -> svmK ')
        print('LogisticRegression() -> logic')