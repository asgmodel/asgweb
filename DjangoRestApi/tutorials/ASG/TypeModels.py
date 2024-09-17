class ModelSetting:
    def __init__(self,name="",path_model="",path_Qwords=""):
        self.name=name
        self.path_model=path_model
        self.path_Qwords=path_Qwords

from BasePath import BASEPATH

TypeModel={
    'group':ModelSetting(name='Group',path_model=BASEPATH+"groupModel_API.pkl",path_Qwords=BASEPATH+"listdesgroupModel_API.pkl"),
    'technique':ModelSetting(name='Technique',path_model=BASEPATH+"TecModel_API.pkl",path_Qwords=BASEPATH+"listdesTecModel_API.pkl"),
    'software':ModelSetting(name='Software',path_model=BASEPATH+"sofwareModel_API.pkl",path_Qwords=BASEPATH+"listdessofwareModel_API.pkl")
}



FormModels={TypeModel['group'].name:
             {'name':'FormGroupS',
               'coldata' : [
                            {"text": "num", "stretch": False,"width":50},

                            "Tactic Name"
                            ,
                            "Technique Name",
                            "Group Name",
                            "Score "
                            ]
             ,
             'msgs':['''• Refers to the search for the highest technique similar to the nature of the specific input.''',
                  '''• Determining the threshold, which is a value between 0 and 1 that allows the width of more than one technique that can be similar to the nature of the input.  ''',
                  ''' Techniques can be searched in two ways''',
                  '''The table shows each technique with the tactic to which it belongs and the score that determines the good prediction probability value in percentage. ''',
                  '''The interface allows the user to enter a description of a technique or a group of techniques that serve the same purpose. The user can also enter the program that uses the technique and predicts the most appropriate technique.'''
                 ]
               ,'placeholder':' , softwares and groups '
             },
            #2
            TypeModel['technique'].name:
             {'name':'FormTechniqueS',
               'coldata' : [
                            {"text": "num", "stretch": False,"width":50},

                            "Tactic Name"
                            ,
                            "Technique Name",

                            "Score "
                            ]
             ,
             'msgs':['''• Refers to the search for the highest technique similar to the nature of the specific input.''',
                  '''• Determining the threshold, which is a value between 0 and 1 that allows the width of more than one technique that can be similar to the nature of the input.  ''',
                  ''' Techniques can be searched in two ways''',
                  '''The table shows each technique with the tactic to which it belongs and the score that determines the good prediction probability value in percentage. ''',
                  '''The interface allows the user to enter a description of a technique or a group of techniques that serve the same purpose. The user can also enter the program that uses the technique and predicts the most appropriate technique.'''
                 ]
              ,'placeholder':'.'
             }



         #3
            ,TypeModel['software'].name:
             {'name':'FormTechniqueS',
               'coldata' : [
                        {"text": "num", "stretch": False,"width":50},
                        "Software Name"
                        ,
                        "Score "
                    ]
             ,
             'msgs':['''• Refers to the search for the highest program similar to the nature of the selected input text.''',
                  '''• Setting the threshold, which is a value between 0 and 1 that allows the display of more than one program that can be similar to the nature of the text entered. ''',
                 '''The program can be searched in two ways ''',
                 '''The table shows the name of the program and the score that determines the probability value of a good forecast, which is a percentage value''',
                 '''The interface allows the user to enter a description of a program or a group of programs within the MITER ATT & CK that serves the same purpose. The user can also enter techniques that may be within the series of program scenarios.''']

             ,'placeholder':' and softwares'
             },

    }