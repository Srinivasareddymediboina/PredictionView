# Generated by Django 3.1.6 on 2021-02-24 06:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('cricanalysisapp', '0003_auto_20210222_1550'),
    ]

    operations = [
        migrations.AlterField(
            model_name='desig',
            name='gender',
            field=models.CharField(max_length=20, null=True),
        ),
    ]