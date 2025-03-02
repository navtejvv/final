import os
import base64
import cv2
import numpy as np
from django.shortcuts import render, redirect
from django.http import JsonResponse
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from django.contrib.auth import authenticate, login

# Paths to pre-trained model and Haar Cascade
MODEL_PATH = "D:\mainproject\project\model.h5"  # Update with actual path
HAAR_CASCADE_PATH = "D:\mainproject\project\haarcascade_frontalface_default.xml"

# Load the model and face classifier
try:
    classifier = load_model(MODEL_PATH)
    face_classifier = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
except Exception as e:
    print(f"Error loading model or Haar Cascade: {e}")

# Emotion labels corresponding to the model's output
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


# LOGIN VIEW
def login_view(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('emotion_detection')
        else:
            return render(request, 'login.html', {'error': 'Invalid username or password'})
    return render(request, 'login.html')


# MAIN EMOTION DETECTION PAGE
def emotion_detection_view(request):
    return render(request, 'emotion_detection.html')


# WEBCAM EMOTION PAGE
def webcam_emotion_view(request):
    return render(request, 'webcam_emotion.html')


# SCREENSHOT PAGE
from collections import Counter

emotion_to_emoji = {
    'Angry': '😠',
    'Disgust': '🤢',
    'Fear': '😨',
    'Happy': '😄',
    'Neutral': '😐',
    'Sad': '😢',
    'Surprise': '😲',
}

import os
from django.conf import settings
from django.shortcuts import render


from collections import Counter

emotion_to_emoji = {
    'Angry': '😠',
    'Disgust': '🤢',
    'Fear': '😨',
    'Happy': '😄',
    'Neutral': '😐',
    'Sad': '😢',
    'Surprise': '😲',
}

import os
from django.shortcuts import render

from collections import Counter

emotion_to_emoji = {
    'Angry': '😠',
    'Disgust': '🤢',
    'Fear': '😨',
    'Happy': '😄',
    'Neutral': '😐',
    'Sad': '😢',
    'Surprise': '😲',
}

from collections import Counter

def screenshot_page_view(request):
    """
    Render the page to display the average emotion detected in the last 10 seconds.
    """
    captured_emotions = request.session.get('captured_emotions', [])  # Retrieve emotions from session

    if captured_emotions:
        # Calculate the most frequently detected emotion
        emotion_counts = Counter(captured_emotions)
        average_emotion = emotion_counts.most_common(1)[0][0]  # Most common emotion
        emoji = emotion_to_emoji.get(average_emotion, '😐')  # Get the emoji for the average emotion
    else:
        average_emotion = "No emotions detected"
        emoji = "😐"

    return render(request, 'screenshot_page.html', {
        'average_emotion': average_emotion,
        'emoji': emoji
    })





def store_emotion(emotion):
    # Assuming you are storing emotions in the session
    session_emotions = request.session.get('captured_emotions', [])
    session_emotions.append(emotion)
    request.session['captured_emotions'] = session_emotions[-10:]  # Keep the last 10 seconds of emotions
    


# PREDICT EMOTION
def predict_emotion(request):
    if request.method == 'POST':
        try:
            img_data = request.POST.get('image_data')
            if not img_data:
                return JsonResponse({'error': 'No image data provided'}, status=400)

            img_data = img_data.split(',')[1]
            img_bytes = base64.b64decode(img_data)
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces) == 0:
                return JsonResponse({'emotion': 'No face detected'})

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float32') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    prediction = classifier.predict(roi)[0]
                    label = emotion_labels[prediction.argmax()]
                    print(f"Predicted emotion: {label}")  # Added debug
                    if 'captured_emotions' not in request.session:
                        request.session['captured_emotions'] = []
                    request.session['captured_emotions'].append(label)
                    request.session.modified = True  # Ensure session saves
                    print(f"Session emotions: {request.session['captured_emotions']}")  # Added debug
                    return JsonResponse({'emotion': label})

            return JsonResponse({'emotion': 'No face detected'})
        except Exception as e:
            print(f"Error in predict_emotion: {e}")  # Enhanced error logging
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=405)

# CAPTURE EMOTION IMAGE
def capture_emotion(request):
    return predict_emotion(request)


# RESTART EMOTION DETECTION
def restart_emotion_detection(request):
    request.session.pop('captured_emotions', None)  # Clear session data
    return redirect('webcam_emotion')


import os
from datetime import datetime

import base64
import os
from django.http import JsonResponse
from django.conf import settings

import os
from datetime import datetime

import base64
import os
from django.http import JsonResponse

import os
from datetime import datetime

def capture_and_save_image(request):
    """
    Capture the image from the frontend, save it locally, and return the path or confirmation.
    """
    if request.method == 'POST':
        try:
            img_data = request.POST.get('image_data')
            if not img_data:
                return JsonResponse({'error': 'No image data provided'}, status=400)

            # Decode Base64 image data
            img_data = img_data.split(',')[1]  # Remove the 'data:image/png;base64,' prefix
            img_bytes = base64.b64decode(img_data)
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Save the image locally
            save_directory = "D:\mainproject\project\EmotionImages"  # Set a directory path to save images
            os.makedirs(save_directory, exist_ok=True)  # Create directory if it doesn't exist
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(save_directory, f"emotion_capture_{timestamp}.png")
            cv2.imwrite(file_path, img)

            return JsonResponse({'message': 'Image saved successfully!', 'file_path': file_path})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)


import os
from django.http import JsonResponse
from django.shortcuts import render

from django.conf import settings
import os

def get_last_image(request):
    directory = settings.MEDIA_ROOT  # Use MEDIA_ROOT path
    
    # Get all image files sorted by modification time
    image_files = sorted(
        (os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))),
        key=os.path.getmtime,
        reverse=True
    )
    
    if image_files:
        last_image_path = image_files[0]
        # Create a URL for the image
        relative_path = os.path.relpath(last_image_path, start=settings.MEDIA_ROOT)
        image_url = os.path.join(settings.MEDIA_URL, relative_path).replace("\\", "/")
        return render(request, 'show_image.html', {'image_url': image_url})
    else:
        return JsonResponse({'error': 'No images found'}, status=404)


from django.shortcuts import render

def show_image(request):
    # Example logic to fetch the image URL
    image_url = "D:\mainproject\project\EmotionImages" # Double backslashes
 # Replace with actual image path logic
    return render(request, 'show_image.html', {'image_url': image_url})


from django.shortcuts import render

# Emotion to Quote Mapping
from django.shortcuts import render
from collections import Counter

# Emotion to Quote Mapping
import os
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse


# Running Emotions Page (Updated)
import os
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse



import os
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse



import os
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse

# Dictionary for emotion quotes
import os
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse

# Dictionary for emotion quotes
emotion_quotes = {
    'Angry': "Anger is one letter short of danger.",
    'Happy': "Happiness is not something ready-made. It comes from your own actions.",
    'Sad': "Tears come from the heart and not from the brain.",
    'Neutral': "Sometimes, the best way to be happy is to stop being so hard on yourself.",
    'Fear': "Fear is a reaction. Courage is a decision.",
    'Surprise': "Surprise is the greatest gift which life can grant us.",
    'Disgust': "Disgust is the feeling of a man who has been bitten by a mosquito who has already been killed."
}

def running_emotions_view(request):
    """
    Render the page with the last image and the quote for the latest detected emotion.
    """
    # Retrieve the last detected emotion from the session
    captured_emotions = request.session.get('captured_emotions', [])
    latest_emotion = captured_emotions[-1] if captured_emotions else "neutral"

    print("Captured Emotions:", captured_emotions)  # Debugging
    print("Latest Emotion Retrieved:", latest_emotion)  # Debugging

    # Get the last image from the MEDIA_ROOT directory
    directory = settings.MEDIA_ROOT  

    try:
        image_files = sorted(
            (os.path.join(directory, f) for f in os.listdir(directory) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))),
            key=os.path.getmtime,
            reverse=True
        )
        if image_files:
            last_image_path = image_files[0]
            relative_path = os.path.relpath(last_image_path, start=settings.MEDIA_ROOT)
            last_image_url = os.path.join(settings.MEDIA_URL, relative_path).replace("\\", "/")
        else:
            last_image_url = None  # No images found
    except Exception as e:
        print(f"Error retrieving last image: {e}")  # Debugging
        last_image_url = None

    # Get the quote for the detected emotion
    quote = emotion_quotes.get(latest_emotion, "No quote available.")

    print("Displaying Quote:", quote)  # Debugging

    return render(request, 'running_emotions.html', {
        'latest_emotion': latest_emotion,
        'quote': quote,
        'last_image_url': last_image_url
    })

import csv
from collections import Counter
from datetime import datetime
from django.http import HttpResponse

import csv
from collections import Counter
from datetime import datetime
from django.http import HttpResponse


def some_view_function(request):
    captured_emotions = request.session.get('captured_emotions', [])
    print("Captured emotions:", captured_emotions)

    if captured_emotions:
        emotion_counts = Counter(captured_emotions)
        average_emotion = emotion_counts.most_common(1)[0][0]
    else:
        average_emotion = "No emotions detected"
        print("No emotions to write to CSV")

    # Set CSV file path to a subdirectory in the project
    csv_file_path = os.path.join(os.path.dirname(__file__), "emotions", "final_emotions.csv")
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)  # Create 'emotions' folder if it doesn’t exist
    print(f"Attempting to write to: {csv_file_path}")

    try:
        file_exists = os.path.isfile(csv_file_path)
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Timestamp", "Emotion"])  # Add headers if file is new
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), average_emotion])
        print(f"Successfully wrote '{average_emotion}' to {csv_file_path}")
    except PermissionError:
        print(f"Permission denied to write to {csv_file_path}")
        return HttpResponse("Permission denied when writing to CSV.", status=500)
    except Exception as e:
        print(f"Error writing to CSV: {e}")
        return HttpResponse(f"Error writing to CSV: {e}", status=500)

    return HttpResponse(f"Emotion '{average_emotion}' saved successfully.")