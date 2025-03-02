from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('emotion_detection/', views.emotion_detection_view, name='emotion_detection'),
    path('webcam_emotion/', views.webcam_emotion_view, name='webcam_emotion'),
    path('predict_emotion/', views.predict_emotion, name='predict_emotion'),
    path('capture_emotion/', views.capture_emotion, name='capture_emotion'),
    path('webcam_capture/', views.webcam_emotion_view, name='webcam_capture'),
    path('screenshot_page/', views.screenshot_page_view, name='screenshot_page'),
    path('restart_emotion_detection/', views.restart_emotion_detection, name='restart_emotion_detection'),
    path('capture_and_save_image/', views.capture_and_save_image, name='capture_and_save_image'),
    path('show_image/', views.get_last_image, name='show_image_page'),
    path('running_emotions/', views.running_emotions_view, name='running_emotions'),
    path('some_view_function/', views.some_view_function, name='some_view_function'),
   

]
