from django.forms import ModelForm
from .models import *
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class ParametersForm(ModelForm):
	class Meta:
		model = Parameters
		fields = '__all__'



class UserSignUpForm(UserCreationForm):
	class Meta:
		model=User
		fields=['first_name','last_name','username','password1','password2','email']

class user_desig(ModelForm):
	class Meta:
		model=Desig
		fields='__all__'


