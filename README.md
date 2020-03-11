# Image-Classifier
A simple web app to predict images using Pytorch and Django

This classifier can predict 3 basic shapes which are circle, square and triangle. This classifier was trained using a pretrained resnet50 on a total of 900 images


To run this project. Follow the following steps
* Clone this repository

* Change working directory to project directory

* Create a virtual environment using this command
! virtualenv venv

* Activate the virtual environment
! source venv/bin/activate

* Install the project requirements
! pip install -r requirements.txt

* Migrate
! python manage.py makemigrations
! python manage.py migrate

* Run server
! python manage.py runserver
