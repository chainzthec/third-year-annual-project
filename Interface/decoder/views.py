from io import BytesIO

from django.shortcuts import render
from django.http import JsonResponse

import application
from application.settings import BASE_DIR
from decoder.decoder import launchTraitment


def home(request):
    return render(request, 'index.html', locals())


def upload(request):
    if request.is_ajax():
        image = request.FILES.get("image")
        modelName = request.POST.get("modelName")

        typeisCorrect = image.content_type in application.settings.VALID_TYPES

        if image and typeisCorrect:

            try:
                return JsonResponse(launchTraitment(image, modelName))

            except Exception as e:
                # return JsonResponse({"error": e})
                return JsonResponse({"error": "Erreur ! Impossible de redimensionner l'image !"}, status=500)
        else:
            return JsonResponse({"error": "Erreur ! Type d'image non supportée !"}, status=500)
    else:
        return JsonResponse({"error": "Erreur ! not xmlhttprequest"}, status=500)



