# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 22:13:12 2022

@author: amaer
"""

import numpy as np



import copy
from ModelTEC import *

import pickle

from  threading import Thread
class Graph:

    def __init__(self,root,name=None,idtactic=0):
        self.Root=root
        self.Name=name
        self.Idtactic=idtactic

        self.Tactics=[]
    #........................

    def Filter(self):
        for i in range(self.Idtactic,len(Scenoariotree.Basetree)):
            self.Tactics.append(copy.deepcopy(Scenoariotree.Basetree[i]))

    #........................

    def Run(self,path="",state=" ",Cinput="",listAttackGod=[],numstop=-1):

        self.Filter()
        for tactic in self.Tactics:
            tactic.Run(self.Root,path=path+"=> "+tactic.Name,state=state,Cinput=Cinput,listAttackGod=listAttackGod,numstop=numstop)
#             thread=Thread(target = tactic.Run, args = (self.Root,path+"->"+tactic.Name))
#             thread.start()
#             thread.join()


    #........................


class Technique:

    PreprocessingFunc=None
    ExtractFeaturesFunc=None
    objFile=None
    Counter=0
    Cinput=' '
    listAttackGod=[]
    PrintScenoario=print



    def __init__(self,name,description,weight=None):

        self.Name=name
        self.Description=description
        self.Weight=weight
        self.CleanDescription=None
        self.FeaturesDescription=None
        self.Graph=None

    #........................................

    def Preprocessing(self):
        if Technique.PreprocessingFunc:
            return Technique.PreprocessingFunc([self.Description])
        return self.Description

    #........................................

    def ExtractFeatures(self):
        if Technique.ExtractFeaturesFunc:
            return Technique.ExtractFeaturesFunc(self.CleanDescription if self.CleanDescription else self.Description)
        return np.zeros((1,10))

    #........................................

    def Fit(self):
        self.CleanDescription=self.Preprocessing()
        #self.FeaturesDescription=self.ExtractFeatures()

    #........................................

    def Run(self,id,path="",state=" ",Cinput="",listAttackGod=[],numstop=-1):

        if (numstop ==-1 or len(listAttackGod)<numstop):
            if id<len(Scenoariotree.Basetree) :
               # Technique.objFile.write("=")

               # print('=',end='')

                path=path+" *-> "+self.Name+" "
                row=Technique.PrintScenoario(path,Cinput)
                if row is not None:
                    listAttackGod.append(row)

                self.Graph=Graph(root=self,name=self.Name,idtactic=id+1)
                self.Graph.Run(path=path,state=state,Cinput=Cinput,listAttackGod=listAttackGod,numstop=numstop)
        else :
            try:
                row=Technique.PrintScenoario(path,Cinput)
                if row is not None:
                    listAttackGod.append(row)

            except:
                pass


    #........................................

    def __str__(self):
        return f"(Name: {self.Name} , Weights: {self.Weight})"



class Tactic():
    dataMS=None
    WF="thrshold"



    def __init__(self,name,id=0,description=None,weight=None,techniques=None,wfilter="thrshold"):
        self.Name=name
        self.Id=id
        self.Weight=weight
        self.Techniques=techniques if techniques else []


    #........................................

    def Filter1(self,root,path=""):
        i=0
        while i<len(self.Techniques):

            sim=Tactic.dataMS[root.Name][self.Techniques[i].Name]

            if sim>=Scenoariotree.ThresholdTechnique:
                self.Techniques[i].Weight=sim


            else:

#                 row=Technique.PrintScenoario(path,Technique.Cinput)
#                 if row is not None:
#                     Technique.listAttackGod.append(row)
#                     Technique.Counter+=1
                self.Techniques.remove(self.Techniques[i])
                i-=1



            i+=1

    #........................................
    def Filter2(self,root,path=""):

        listd=[]
        max=0
        for i in range(len(self.Techniques)):
            sim=Tactic.dataMS[root.Name][self.Techniques[i].Name]
            if sim>max and  sim>=Scenoariotree.ThresholdTechnique:
                listd=[self.Techniques[i]]
                max=sim
#             Scenoariotree.CurrentSoft.predict()


        self.Techniques=listd

    #........................................
    def Filter(self,root,path=""):
        if Tactic.WF=="thrshold":
            self.Filter1(root,path)
        else:
            self.Filter2(root,path)


    def Run(self,root,path="",state="",Cinput="",listAttackGod=[],numstop=-1):

        self.Filter(root,path)

        for technique in self.Techniques:
            technique.Run(self.Id,path=path,state=state,Cinput=Cinput,listAttackGod=listAttackGod,numstop=numstop)
#             thread=Thread(target=technique.Run,args=(self.Id,path))
#             thread.start()
#             thread.join()

    #........................................

    def __str__(self):
        return f"(Name: {self.Name}, Weight: {self.Weight})"



from BasePath import BASEPATH

class Scenoariotree:

    obTEC=None
    obTECSodft=None
    ThresholdTechnique=0.3
    ThresholdTactic=0.3
    Basetree=None
    CurrentSoft=None
    indecisT={}
    CurrentInput=''


    #....................................................







    #....................................................


    def __init__(self,file1,file2,Ntactics=None):

        self.TacticsOrder=pickle.load(open(BASEPATH+file1,'rb'))
        self.indecisT=pickle.load(open(BASEPATH+file2,'rb'))
        self.Ntactics=len(self.TacticsOrder)
        self.nlp=None#spacy.load('en_core_web_lg')
        """

        """

    def setModels(self,ObTEC=None,ObTECSodft=None,obchk=None,isForm=False):
        self.obTEC=ObTEC
        self.obTECSodft=ObTECSodft
        self.obTEC.ChangeModel(ob="svmK")
        self.obTECSodft.ChangeModel(ob="svmK")
        self.obchk=obchk
        Technique.PrintScenoario=obchk.print_table if isForm==False else obchk.process

    #....................................................

    def InitBasetree(self):
        tree=[]
        i=0
        for key  in self.TacticsOrder:
            tactic=key
            tree.append(Tactic(name=tactic,id=i))
            group=self.TacticsOrder[key]
            i+=1
            for j in range(len(group)):
                tree[-1].Techniques.append(Technique(name=group[j],
                                                     description=" ",
                                                     ))


        return tree

    #..............................................

    def getordertactics(self):
        return Scenoariotree.Basetree
    def Fit(self,file="dataSM.pkl",wtype='ALL'):





        Tactic.dataMS=pickle.load(open(BASEPATH+file,'rb'))
        Scenoariotree.Basetree=self.InitBasetree()


#        Scenoariotree.Basetree[6].Techniques.append(Technique(name="Exploitation of Remote Services",description="Exploitation of Remote Services")



    #..............................................

    def Predict(self,Description,ThresholdTechnique=0.5,ThresholdTactic=0.0,WF="thrshold",istrans=False,numstop=-1):
        Technique.Counter=0
        Technique.listAttackGod=[]
        self.obchk.inintlist()
        Tactic.WF=WF
        Scenoariotree.ThresholdTechnique=ThresholdTechnique
        Scenoariotree.ThresholdTactic=ThresholdTactic
        if istrans:
            Description=[self.obTEC.to_tran(Description)]

        tech=self.obTEC.predict(Description)

        if tech!="No":
            print('-------------------------------------------')
            print('Technique : ', tech)
            tact=self.indecisT[tech]

            print(' Tactic :',tact)
            self.obTECSodft.ChangeModel(ob="svmK")
            CurrentSoft=self.obTECSodft.predict(Description+[" "+tech])

            print('input  same as Software is  : ',CurrentSoft)
            self.obTECSodft.ChangeModel(ob="svmL")
            print('-------------------------------------------')
            print('---------------------senarios----------------------')
            listofs=[]

            self.Root=Technique(name=tech,description=Description)

            idt=-1
            for key in self.TacticsOrder:
                if key==tact:
                    break
                idt+=1

            self.Root.Run(id=idt+1,path="=>  "+tact,state="",Cinput=Description[0],listAttackGod=listofs,numstop=numstop)
            if idt==self.Ntactics-1:
                self.obchk.process(tact+" *-> "+ tech,Description[0],isclean=False)








        else :
            print( "no found  any thing........")

        return listofs
    #..............................................




