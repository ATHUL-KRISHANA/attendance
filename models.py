import os
import cv2
import pickle
import datetime
from mtcnn import MTCNN
from django.db import models
from keras_facenet import FaceNet
from django.dispatch import receiver
from django.db.models.signals import post_save
from django.contrib.auth.models import User



mtcnn = MTCNN()
embedder = FaceNet()


data_dir = 'media'
pickle_dir = 'enrollment'


# Stack model-------------------------------------------------------------------

class Stack(models.Model):
    stack_name = models.CharField(max_length=20)
    description = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.stack_name



# Faculty model-----------------------------------------------------------------

class Faculty(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    first_name = models.CharField(max_length=70)
    last_name = models.CharField(max_length=70)  
    join_date = models.DateField()  
    image = models.ImageField(upload_to='faculty_faces/') 

    def __str__(self):
        return f'{self.first_name} {self.last_name}'

# student model ----------------------------------------------------------------

class Student(models.Model):
    first_name = models.CharField(max_length=70)
    last_name = models.CharField(max_length=70)
    date_of_birth = models.DateField()
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    
    stack = models.ForeignKey(Stack, on_delete=models.CASCADE,null=True,blank=True)

    MONTH_CHOICES = [
        ('January', 'January'),
        ('February', 'February'),
        ('March', 'March'),
        ('April', 'April'),
        ('May', 'May'),
        ('June', 'June'),
        ('July', 'July'),
        ('August', 'August'),
        ('September', 'September'),
        ('October', 'October'),
        ('November', 'November'),
        ('December', 'December'),
    ]
    batch = models.CharField(max_length=20, choices=MONTH_CHOICES)
    phone = models.BigIntegerField()
    email = models.EmailField()
    address = models.TextField(null=True, blank=True)

    STATUS_CHOICES = [
        ('Present', 'Present'),
        ('Absent', 'Absent'),
        ('Interview', 'Interview'),
        ('Holiday', 'Holiday'),
    ]
    present = models.CharField(max_length=10, choices=STATUS_CHOICES, default='Absent')
    join_date = models.DateField(default=datetime.date.today)
    image = models.ImageField(upload_to="media")

    def __str__(self):
        return f'{self.first_name} {self.last_name}'

# attendence  --------------------------------------------------------------------------
   
class AttendanceRecord(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    date = models.DateField(default=datetime.date.today)
    status = models.CharField(max_length=10, choices=Student.STATUS_CHOICES, default='Absent')

    class Meta:
        unique_together = ('student', 'date')

    def __str__(self):
        return f'{self.student} - {self.date} - {self.status}'


#-------------------------------------------------------------------------------------

if not os.path.exists(pickle_dir):
    os.makedirs(pickle_dir)


def preprocess_and_extract_embeddings(image_path):
    embeddings = []
    if image_path.endswith(('.jpg', '.jpeg')):
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = mtcnn.detect_faces(rgb_image)
        if detections:
            for detection in detections:
                x, y, w, h = detection['box']
                face = rgb_image[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (160, 160))
                embedding = embedder.embeddings([face_resized])[0]
                embeddings.append(embedding)
    return embeddings


def save_embeddings(embeddings, name):
    file_path = os.path.join(pickle_dir, f'{name}_embeddings.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)



@receiver(post_save, sender=Student)
def generate_embeddings(sender, instance, created, **kwargs):
    if created: 
        image_path = instance.image.path
        name = f'{instance.first_name}_{instance.last_name}'
        embeddings = preprocess_and_extract_embeddings(image_path)
        save_embeddings(embeddings, name)


@receiver(post_save, sender=Faculty)
def generate_embeddings(sender, instance, created, **kwargs):
    if created: 
        image_path = instance.image.path
        name = f'{instance.first_name}_{instance.last_name}'
        embeddings = preprocess_and_extract_embeddings(image_path)
        save_embeddings(embeddings, name)
#---------------------------------------------------------------------------------

