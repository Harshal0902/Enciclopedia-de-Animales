from django.shortcuts import render,redirect
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.models import User
from .models import modelimg
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
from Machine_Learning_Model.utilize_model import classify_image

def home(request):
    data = modelimg.objects.all()
    context = {
        'data':data
    }
    return render(request,"home.html",context=context)

def test(request):
    if request.method=='POST':
        if request.FILES['myfile']:
            myfile = request.FILES['myfile']
            fs = FileSystemStorage(location='media')
            filename = fs.save(myfile.name, myfile)
            url = str(BASE_DIR)+'\\media'+'\\'+filename
            res = classify_image(url)
            c=res['class']
            d = res['probablity']
            img = modelimg()
            img.img = filename
            img.predicton = c
            img.probability = d
            img.save()
            return redirect('dashboard')
    return render(request,"test.html")

def explore(request):
    return render(request,"explore.html")