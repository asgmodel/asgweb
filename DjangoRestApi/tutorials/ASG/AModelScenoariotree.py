# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 23:13:18 2023

@author:
"""
from ModelTEC import *
from  MScenoariotree  import *
from  MCheakscenario  import Cheakscenario



class ModelScenoariotree:

    def __init__(self,obTECSoft=None,obTEC=None,obG=None,Base=None,isForm=False):


        self.cks=Cheakscenario(ob=obTECSoft)
#         obTEC.nlp=Base.nlp
#         obTECSoft.nlp=Base.nlp
        self.Base=Base
        self.obGroup=obG
        self.Base.setModels(ObTEC=obTEC,ObTECSodft=obTECSoft,obchk=self.cks,isForm=isForm)


    def search(self,inputstate=[''],rateerror=0,PrintFuri=True,type_search='Max',ThresholdTechnique=0.0,istrans=False):

        Cheakscenario.PrintFuri=PrintFuri
        Cheakscenario.rateerror=rateerror

        self.Base.Predict(Description=inputstate,WF=type_search,ThresholdTechnique=ThresholdTechnique,istrans=istrans)
        print ('all  scenario : end')








