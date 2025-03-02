from django.db import models

class DetectedEmotion(models.Model):
    username = models.CharField(max_length=100)
    emotion = models.CharField(max_length=50)
    timestamp = models.DateTimeField(auto_now_add=True)


from django.db import models

class CapturedImage(models.Model):
    image_data = models.TextField()  # To store the base64 image
    created_at = models.DateTimeField(auto_now_add=True)  # To track when the image was saved

    def __str__(self):
        return f"Captured Image {self.id} - {self.created_at}"

