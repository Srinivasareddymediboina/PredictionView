from django.db import models
from django.contrib.auth.models import User
# Create your models here.
class Parameters(models.Model):
	TEAM_CHOICES = (

		('1','Mumbai Indians'),
		('2','Kolkata Knight Riders'),
		('3','Royal Challengers Bangalore'),
		('4','Deccan Chargers'),
		('5','Chennai Super Kings'),
		('6','Rajasthan Royals'),
		('7','Delhi Daredevils'),
		('8','Gujarat Lions'),
		('9','Kings XI Punjab'),
		('10','Sunrisers Hyderabad'),
		('11','Rising Pune Supergiants'),
		('12','Kochi Tuskers Kerala'),
		('13','Pune Warriors'),
	)
	CITY_CHOICES = (
		('2','Bangalore'),
		('7','Chandigarh'),
		('10','Delhi'),
		('24','Mumbai'),
		('22','Kolkata'),
		('17','Jaipur'),
		('15','Hyderabad'),
		('8','Chennai'),
		('5','Cape Town'),
		('26','Port Elizabeth'),
		('13','Durban'),
		('6','Centurion'),
		('14','East London'),
		('18','Johannesburg'),
		('20','Kimberley'),
		('4','Bloemfontein'),
		('1','Ahmedabad'),
		('9','Cuttack'),
		('25','Nagpur'),
		('11','Dharamsala'),
		('21','Kochi'),
		('16','Indore'),
		('32','Visakhapatnam'),
		('27','Pune'),
		('28','Raipur'),
		('30','Ranchi'),
		('0','Abu Dhabi'),
		('31','Sharjah'),
		('12','Dubai'),
		('29','Rajkot'),
		('19','Kanpur'),
		('23','Mohali'),
		('3','Bengaluru'),
	)
	VENUE_CHOICES = (
		('14','M Chinnaswamy Stadium'),
		('22','Punjab Cricket Association Stadium, Mohali'),
		('8','Feroz Shah Kotla'),
		('34','Wankhede Stadium'),
		('7','Eden Gardens'),
		('26','Sawai Mansingh Stadium'),
		('23','Rajiv Gandhi International Stadium, Uppal'),
		('15','MA Chidambaram Stadium, Chepauk'),
		('4','Dr DY Patil Sports Academy'),
		('19','Newlands'),
		('30','St Georges Park'),
		('13','Kingsmead'),
		('32','SuperSport Park'),
		('2','Buffalo Park'),
		('18','New Wanderers Stadium'),
		('3','De Beers Diamond Oval'),
		('20','OUTsurance Oval'),
		('1','Brabourne Stadium'),
		('24','Sardar Patel Stadium, Motera'),
		('0','Barabati Stadium'),
		('33','Vidarbha Cricket Association Stadium, Jamtha'),
		('10','Himachal Pradesh Cricket Association Stadium'),
		('17','Nehru Stadium'),
		('11','Holkar Cricket Stadium'),
		('5','Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium'),
		('31','Subrata Roy Sahara Stadium'),
		('27','Shaheed Veer Narayan Singh International Stadium'),
		('12','JSCA International Stadium Complex'),
		('29','Sheikh Zayed Stadium'),
		('28','Sharjah Cricket Stadium'),
		('6','Dubai International Cricket Stadium'),
		('16','Maharashtra Cricket Association Stadium'),
		('21','Punjab Cricket Association IS Bindra Stadium, Mohali'),
		('25','Saurashtra Cricket Association Stadium'),
		('9','Green Park'),
	)
	TOSS_CHOICES = (
		('1','Field'),
		('2','Bat'),
	)

	team1 = models.CharField(max_length=5,choices = TEAM_CHOICES)
	team2 = models.CharField(max_length=5,choices = TEAM_CHOICES)
	city = models.CharField(max_length=5,choices = CITY_CHOICES)
	toss_decision = models.CharField(max_length=1,choices = TOSS_CHOICES)
	toss_winner = models.CharField(max_length=5,choices = TEAM_CHOICES)
	venue = models.CharField(max_length=5,choices = VENUE_CHOICES)

class Desig(models.Model):
	#options=[('Male','Male'),('Female','Female')]
	dob=models.DateField(null=True)
	gender=models.CharField(null=True,max_length=20)
	photo=models.ImageField(null=True,upload_to='poster/')
	user=models.OneToOneField(User,on_delete=models.CASCADE)