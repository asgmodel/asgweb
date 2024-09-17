from django.urls import re_path as url
from tutorials import views


urlpatterns = [

    url(r'^api/input/(?P<pk>.+)$', views.input),
    url(r'^api/search/(?P<pk>.+)$', views.search),
     url(r'^api/searchall/(?P<pk>.+)$', views.searchall),
     url(r'^api/search2/(?P<pk>.+)$', views.search2),
    url(r'^api/inputpipline/(?P<pk>.+)$', views.inputpipline),
    url(r'^api/inputgroup/(?P<pk>.+)$', views.input_group),
    url(r'^api/inputgroupinfo/(?P<pk>.+)$', views.input_info_Group),
     url(r'^api/inputsoftinfo/(?P<pk>.+)$', views.input_info_Soft),
    url(r'^api/inputsofware/(?P<pk>.+)$', views.input_soft),
     url(r'^api/transe/(?P<pk>.+)$', views.transe),
     url(r'^api/nlpto/(?P<pk>.+)$', views.nlpto),
      url(r'^api/tecbytac/(?P<pk>.+)$', views.getteck),
       url(r'^api/generatingai/(?P<pk>.+)$', views.generatingai),
       url(r'^api/vits/(?P<pk>.+)$', views.get_answer_ai2),
        url('stramvits/(?P<pk>.+)$',views.get_answer_ai2),
          url('',views.redirect_view),

]