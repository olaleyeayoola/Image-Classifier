# Import Django Libraries and Paths
from django.shortcuts import render, redirect
from image_classifier.settings import DATA_PATH, MEDIA_ROOT
from .models import Search
from .forms import SearchForm
from django.http import HttpResponse
import json
# Import PyTorch libraries
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from torch.autograd import Variable


class_names = ['circle', 'square','traingle']
path = DATA_PATH+'/test.pth'
model_resnet = torchvision.models.resnet50(pretrained=True).to("cpu")
model_resnet = torch.load(path)
model_resnet.eval()
loader = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])



def image_loader(image_name):
    """load image, returns tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0) 
    return image


def predict(classifer, image):
    predictions = classifer(image)
    predicted_classes = []
    for prediction in predictions.cpu().data.numpy():
        # The prediction for each image is the probability for each class,
        #  e.g. [0.8, 0.1, 0.2]
        # So get the index of the highest probability
        class_idx = np.argmax(prediction)
        # Add add the corresponding class name to the results
        predicted_classes.append(class_names[class_idx])
    return np.array(predicted_classes)


def search(request):
    if request.method == 'POST':
        form = SearchForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            image_i = form.cleaned_data.get('image')
            name = form.cleaned_data.get('name')
            model = Search.objects.get(name=name)
            image_url = model.image.url
            img_path = MEDIA_ROOT + '/' + model.get_image_name()
            image = image_loader(img_path)
            prediction = predict(model_resnet, image)
            context = {
                'name': name,
                'prediction' : prediction[0]
            }
            with open('file.json', 'w') as json_file:
                json.dump(context, json_file)
            return redirect('result')
    else:
        form = SearchForm()
    return render(request, 'search_app/search_form.html', {'form': form})

def result(request):
    with open('file.json', 'r') as f:
        data = json.load(f)
    name = data['name']
    prediction = data['prediction']
    model = Search.objects.get(name=name)
    context = {
        'model' : model,
        'prediction' : prediction
    }
    return render(request, 'search_app/result.html', context)

def correct(request):
    with open('file.json', 'r') as f:
        data = json.load(f)
    name = data['name']
    Search.objects.get(name=name).delete()
    return render(request, 'search_app/final.html', {'message': "Yuupp, I knew it"})

def wrong(request):
    with open('file.json', 'r') as f:
        data = json.load(f)
    name = data['name']
    model = Search.objects.get(name=name).delete()
    return render(request, 'search_app/final.html', {'message': "My bad, I'll do better next time"})


