from django.urls import path
from . import views

urlpatterns = {
    # path('', views.summary, name='input_text'),
    path('', views.task_summary, name='task_summarys'),
}