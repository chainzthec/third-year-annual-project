from io import BytesIO

from django.shortcuts import render
from django.http import JsonResponse
from PIL import Image

import application
from application.settings import BASE_DIR
from decoder.decoder import launchTraitment


def home(request):
    test = "coucou toi"
    return render(request, 'index.html', locals())


def upload(request):
    if request.is_ajax():
        image = request.FILES.get("image")
        typeisCorrect = image.content_type in application.settings.VALID_TYPES

        if image and typeisCorrect:
            url = BASE_DIR + "/uploads/" + str(image)
            print(url)
            try:
                file = Image.open(image.file)
                file = file.resize(application.settings.IMAGE_SIZE, Image.ANTIALIAS)
                # file.save(url, 'JPEG', quality=90)

                bytesio = BytesIO()
                file.save(bytesio, 'JPEG', quality=90)
                serializedImage = bytesio.getvalue()

                return JsonResponse(launchTraitment(serializedImage))

            except Exception:
                return JsonResponse({"error": "Erreur ! Impossible de redimensionner l'image !"}, status=200)
        else:
            return JsonResponse({"error": "Erreur ! Type d'image non supportée !"}, status=500)
    else:
        return JsonResponse({"error": "Erreur ! not xmlhttprequest"}, status=500)
