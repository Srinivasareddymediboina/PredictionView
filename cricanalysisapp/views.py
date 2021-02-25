from django.shortcuts import render
import numpy as np
from .forms import *
from . import ca
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required


# Create your views here.
#'team1', 'team2', 'venue', 'toss_winner','city','toss_decision'
def home(req):
    return render(req,'home.html')

def register(req):
    if req.method=='POST':
        user=User.objects.create_user(first_name=req.POST['firstname'],
            last_name=req.POST['lastname'],email=req.POST['email'],
            username=req.POST['username'],password=req.POST['password'])
        dob=req.POST['dob']
        gender=req.POST['gender']
        photo=req.FILES['photo']
        Desig.objects.create(dob=dob,photo=photo,gender=gender,user=user)
        messages.success(req,req.POST['username']+' is Succefully Registered')

    form = UserSignUpForm()
    pform=user_desig()
    return render(req,'register.html',{'form':form,'pform':pform})
@login_required
def ParametersView(request):
    submitbutton= request.POST.get("submit")
    form= ParametersForm(request.POST or None)
    output = ''
    inp_arr = ''
    arr = np.arange(6)
    if form.is_valid():
    	team1 = form.cleaned_data['team1']
    	team2 = form.cleaned_data['team2']
    	venue = form.cleaned_data['venue']
    	toss_winner = form.cleaned_data['toss_winner']
    	city = form.cleaned_data['city']
    	toss_decision = form.cleaned_data['toss_decision']

    	arr[0],arr[1],arr[2],arr[3],arr[4],arr[5] = (team1, team2, venue, toss_winner, city, toss_decision)
    	arr = arr.reshape(1, -1)
    	inp_arr = arr
    	output = ca.predict(inp_arr)
    	teams = ['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore',
                 'Deccan Chargers','Chennai Super Kings','Rajasthan Royals',
                 'Delhi Daredevils','Gujarat Lions','Kings XI Punjab','Sunrisers Hyderabad',
                 'Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors']
    	for i in output:
    		output = i
    	output = teams[i-1]
    context= {'form': form, 'output' : output,'submitbutton': submitbutton, 'inp_arr' : inp_arr}
    return render(request, 'index.html', context)

def profile(req,id):
    data=User.objects.get(id=id)
    data1=Desig.objects.get(id=id)

    return render(req,'profile.html',{'i':data,'i2':data1})

def about(req):

    return  render(req,'about.html')

def teams(req):

    return render(req,'teams.html')


