from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .model import predict_image
import os
import uuid

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
from django.http import HttpResponse

def home(request):
    return HttpResponse("Melanoma API is running")
@csrf_exempt
def predict(request):
    if request.method == "POST":
        file = request.FILES.get("file")

        if not file:
            return JsonResponse({"error": "No file"}, status=400)

        filename = str(uuid.uuid4()) + ".jpg"
        path = os.path.join(UPLOAD_DIR, filename)

        with open(path, "wb+") as f:
            for chunk in file.chunks():
                f.write(chunk)

        label, confidence = predict_image(path)

        return JsonResponse({
            "prediction": label,
            "confidence": round(confidence, 4)
        })

    return JsonResponse({"message": "Send POST request"})
