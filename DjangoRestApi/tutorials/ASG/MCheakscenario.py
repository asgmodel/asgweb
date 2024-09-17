# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 21:43:54 2022

@author: amaer
"""
import numpy as np

from ModelTEC import *


class  Cheakscenario:
    rateerror=0
    PrintFuri=True

    def __init__(self,ob=None,sequensed=None):
        self.obTECSoft=ob
        self.lensoft=len(ob.obMP.Classes)
        self.inintlist()
        self.onsequens=sequensed


    def inintlist(self):
        self.listsoftware=np.zeros((self.lensoft,5))
        self.Counter=0



    def  createtable(self,row,stat,index=0):
        st='<table border="1" style="" ><tr  style="background-color:'+stat[0]+'; color:#fff"><th  style="text-align:center" >Tactic  Name</th><th  style="text-align:center" >Technique Name</th></tr>'



        for ob in row:
            st+='<tr><td  style="text-align:center"  >'+ob[0]+'</td><td style="text-align:center"  >'+ob[1]+'</td></tr>'

        st+='<tr style="background-color:'+stat[0]+'; color:#fff"><td colspan = "2" style="text-align:center" >attack scenario status ('

        st+=''+ stat[1] +' )  </td></tr>'
        return st+'</table>'




    def getScenario(self,itmes,k=0,index=0):
        stat=self.getAStatescenario(k,index)
        if Cheakscenario.PrintFuri==False and stat[1]=='Failure':return None
        rowd=[]
        for ob in itmes:
            if ob.strip() !='':

                itms=ob.split("*->")
                rowd.append([itms[0],itms[1]])
        return rowd,stat,k




    def  process(self,path,cinput,isclean=True):
        itmes=path.split("=> ")

        pathT=self.getpathTech(itmes)
        if isclean:
            da=self.obTECSoft.clean_dataT([pathT+' '+str(cinput) +" use  "])
        else:
            da=[cinput]



       # print(pathT,da)
        if(len(da)>0):
            try:
                feat=self.obTECSoft.obMP.obVec.transform(da)
                pyd=self.obTECSoft.Model.predict_proba(feat)
                self.listsoftware[:,4]+=np.array(pyd[0])
               #for i in range(self.lensoft):
              #      self.listsoftware[i,4]=pyd[0][i]
                index=np.argmax(pyd)
                k=pyd[0][index]*100+Cheakscenario.rateerror

                row=self.getScenario(itmes,k,index)
                if row is not None and len(row[0])>0:
                    self.Counter+=1
                    if self.onsequens is not None:
                        self.onsequens((row,pyd[0]))


                    return row,pyd[0]


            except: pass

        return None

    def print_table(self,path,cinput):
       # self.obTECSoft.ChangeModel(ob='svmL')
        rows=self.process(path, cinput)

        if rows is not None :
            row,pa=rows
            td=self.createtable(row[0],row[1],self.Counter)
            print('---------------------------scenario (',self.Counter,')------------------------------')
            # display(HTML(td))

            return rows


    def getAStatescenario(self,k,index):
        if k>=70:
            self.listsoftware[index][0]+=1
            return ('#04AA6D','Success')
        if k>=60:
            self.listsoftware[index][1]+=1
            return ('#1ec7c3','close to success')

        if k>=30:
            self.listsoftware[index][2]+=1
            return ('#fb4bd1','Initi')
        self.listsoftware[index][3]+=1
        return ('#e91c07','Failure')
    def getpathTech(self,items):
        path=' use '
        for ob in items:
             if ob.strip() !='':

                itms=ob.split("*->")
                path+=' '+itms[1]
        return path






