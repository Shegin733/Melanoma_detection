from django.shortcuts import render
from django.http import JsonResponse
from .model import predict_image
import os
import uuid

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


def home(request):
    return render(request, "index.html")


def predict(request):

    if request.method == "POST":

        file = request.FILES.get("file")

        if not file:
            return JsonResponse({
                "error": "No file uploaded"
            })

        filename = str(uuid.uuid4()) + ".jpg"

        path = os.path.join(UPLOAD_DIR, filename)

        with open(path, "wb+") as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        label, confidence = predict_image(path)

        return JsonResponse({
            "prediction": label,
            "confidence": round(confidence, 4)
        })

    return JsonResponse({
        "message": "Send POST request"
    })