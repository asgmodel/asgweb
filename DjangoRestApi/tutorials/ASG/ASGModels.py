from ModelTEC import *

import pickle
from TypeModels import TypeModel
from BasePath import BASEPATH
from threading import Thread
from MScenoariotree import *
from AModelScenoariotree import ModelScenoariotree

from  MCheakscenario  import Cheakscenario
class ASG:
    def readobsoft(self):
        self.Soft=TEC(typemodel="svmL",model_setting=TypeModel['software'])
        self.Soft.loadmodel()
        self.Index+=1

    def readobgroup(self):
        self.Group=TEC(typemodel="svmL",model_setting=TypeModel['group'])
        self.Group.loadmodel()
        self.Index+=1

    def readobtec(self):
        self.Tec=TEC(typemodel="svmL",model_setting=TypeModel['technique'])
        self.Tec.loadmodel()
        self.Index+=1

    def readmstree(self):
        self.SGT=Scenoariotree('TTs.pkl','tecbyTectics.pkl')
        self.SGT.Fit()
        self.Index+=1
    def __init__(self):
        self.Index=0
        self.SGT=None
        # Thread(target=self.readmstree).start()
        # Thread(target=self.readobsoft).start()
        # Thread(target=self.readobtec).start()
        # Thread(target=self.readobgroup).start()
        self.readmstree()
        self.readobsoft()
        self.readobtec()
        self.readobgroup()


        self.Soft.setPipeline(model=self.Tec)
        self.Group.setPipeline(model=self.Soft)
        self.cks=Cheakscenario(ob=self.Soft)
        self.MST=ModelScenoariotree(obTECSoft=self.Soft,obTEC=self.Tec,obG=self.Group,Base=self.SGT,isForm=True)
        self.Tec.obMP
        self.SGT.setModels(ObTEC=self.Tec,ObTECSodft=self.Soft,obchk=self.cks,isForm=False)



    def search(self,inputstate=[''],rateerror=0,PrintFuri=True,type_search='Max',ThresholdTechnique=0.5,istrans=False,numstop=-1):

        Cheakscenario.PrintFuri=PrintFuri
        Cheakscenario.rateerror=rateerror

        outputs=self.SGT.Predict(Description=inputstate,WF=type_search,ThresholdTechnique=ThresholdTechnique,istrans=istrans,numstop=numstop)
        print ('all  scenario : end')
        return outputs

