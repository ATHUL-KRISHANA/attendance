import os
import cv2
import time
import base64 
import pickle
import pandas as pd
from datetime import datetime
from keras_facenet import FaceNet
from yoloface import face_analysis
from django.contrib import messages
from django.contrib.auth import  login
from scipy.spatial.distance import cosine
from .models import Student,Stack,Faculty    
from django.shortcuts import render,redirect
from django.core.files.base import ContentFile
from django.shortcuts import render, get_object_or_404




face = face_analysis()
embedder = FaceNet()

data_dir = 'media'
pickle_path = 'enrollment'


def recognize_face(face_embedding):
        min_distance = float('inf')
        recognized_person = "UNKNOWN"
        recognition_accuracy = 0

        for file in os.listdir(pickle_path):
            if file.endswith('_embeddings.pkl'):
                stored_embeddings = pickle.load(open(os.path.join(pickle_path, file), 'rb'))
                for stored_embedding in stored_embeddings:
                    distance = cosine(face_embedding, stored_embedding)
                    accuracy = (1 - distance) * 100  # Convert cosine distance to accuracy

                    if distance < min_distance:
                        min_distance = distance
                        # Extract name from file name
                        recognized_person = file.split('_')[0] + ' ' + file.split('_')[1]
                        recognition_accuracy = accuracy

        # Return recognized person and accuracy if within threshold
        if min_distance < 0.6:  # Threshold can be adjusted
            return recognized_person, recognition_accuracy
        else:
            return "UNKNOWN", 0

# index page------------------------------------------------------------

def index(request):
    return render(request,"index.html")

# student add page -----------------------------------------------------


def add_student(request):
    if request.method == "POST":
        data=request.POST
        first_name = data.get("first_name")
        last_name = data.get("last_name")
        date_of_birth = data.get("date_of_birth")
        gender = data.get("gender")
        stack = data.get("stack")
        batch = data.get("batch")
        phone = data.get("phone")
        email = data.get("email")
        address = data.get("address")
        join_date = data.get("join_date")
        image = data.get("image") 

        stack = Stack.objects.get(id=stack)
        if image: 
            format, imgstr = image.split(';base64,') 
            ext = format.split('/')[-1] 
            image_name = f"{first_name}{last_name}.jpg"
            image = ContentFile(base64.b64decode(imgstr), name=image_name)

        Student.objects.create(
            first_name=first_name,
            last_name=last_name,
            date_of_birth=date_of_birth,
            gender=gender,
            stack=stack,
            batch=batch,
            phone=phone,
            email=email,
            address=address,
            join_date=join_date,
            image=image            
        )
        messages.success(request, 'Student added successfully') 
        return redirect('add_student')

    context={
        'stacks': Stack.objects.all(),
        'MONTH_CHOICES': Student.MONTH_CHOICES
    }
    return render(request,"add_student.html",context)

# attendence --------------------------------------------------------

def attendance(request):
    if request.method == "GET":
        return render(request, "index.html")

    if request.method == "POST":
        selected_time = int(request.POST.get("time", 1))  # Time in minutes
        duration = selected_time * 60  # Convert to seconds

    start_time = time.time()

    # Video capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, boxes, _ = face.face_detection(frame_arr=frame, frame_status=True, model='tiny')

        if len(boxes) > 0:
            for box in boxes:
                x, y, w, h = box
                face_crop = frame[y:y + w, x:x + h]

                if face_crop is not None and face_crop.size > 0:
                    face_resized = cv2.resize(face_crop, (160, 160))

                    # Get face embedding
                    face_embedding = embedder.embeddings([face_resized])[0]
                    
                    # Recognize face and get accuracy
                    person, accuracy = recognize_face(face_embedding)
                    print('Accuracy:', accuracy)
                    print(f"Recognized: {person} with accuracy: {accuracy:.2f}%")

                    if person != "UNKNOWN" and accuracy > 72:
                        name_parts = person.split()
                        first_name = name_parts[0]
                        last_name = name_parts[1] if len(name_parts) > 1 else ""

                        # Update attendance
                        try:
                            student = get_object_or_404(Student, first_name=first_name, last_name=last_name)
                            student.present = "Present"
                            student.save()
                            print(f"Attendance marked for {first_name} {last_name}")
                        except Student.DoesNotExist:
                            print(f"No student found with the name {first_name} {last_name}")

                        # Display recognized name and accuracy
                        cv2.putText(frame, f"{person} ({accuracy:.2f}%)", (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.rectangle(frame, (x, y), (x + h, y + w), (0, 255, 0), 2)

        # Show the video stream
        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27 or (time.time() - start_time) > duration:
            break

    cap.release()
    cv2.destroyAllWindows()

    return render(request, "index.html")

# faculty login --------------------------------------------------

def faculty_login(request):
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    duration = 30  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, boxes, _ = face.face_detection(frame_arr=frame, frame_status=True, model='tiny')

        if len(boxes) > 0:
            for box in boxes:
                x, y, w, h = box
                face_crop = frame[y:y + w, x:x + h]

                if face_crop is not None and face_crop.size > 0:
                    face_resized = cv2.resize(face_crop, (160, 160))

                    face_embedding = embedder.embeddings([face_resized])[0]
                    
                    person, accuracy = recognize_face(face_embedding)
                    print('Accuracy:', accuracy)
                    print(f"Recognized: {person} with accuracy: {accuracy:.2f}%")

                    if person != "UNKNOWN" and accuracy > 72:
                        name_parts = person.split()
                        first_name = name_parts[0]
                        last_name = name_parts[1] if len(name_parts) > 1 else ""

                        try:
                            faculty = get_object_or_404(Faculty,first_name__iexact=first_name, last_name__iexact=last_name)
                            
                            user = faculty.user
                            if user.is_superuser:
                                login(request, user)                                
                                return redirect('/admin/')  
                            else:
                                print(f"{first_name} {last_name} is not a superuser.")

                        except Faculty.DoesNotExist:
                            print(f"No faculty found with the name {first_name} {last_name}")

                        cv2.putText(frame, f"{person} ({accuracy:.2f}%)", (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.rectangle(frame, (x, y), (x + h, y + w), (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27 or (time.time() - start_time) > duration:
            break

    cap.release()
    cv2.destroyAllWindows()
    return render(request,"index.html")


#-----------------------------------------------------------------------------------------------------------

from datetime import datetime, timedelta
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from .models import AttendanceRecord, Student

def sheet(request):
    today = datetime.today().date()
    last_30_days = [(today - timedelta(days=i)) for i in range(60)]
    earliest_date = min(last_30_days) 

    # Fetch attendance records
    attendance_records = AttendanceRecord.objects.filter(date__gte=earliest_date).select_related('student')

    # Process attendance data
    attendance_data = {}
    for record in attendance_records:
        record_date = record.date if isinstance(record.date, datetime) else record.date
        if record_date not in last_30_days:
            continue  

        student = record.student
        if student not in attendance_data:
            attendance_data[student] = {day: "--" for day in last_30_days}
        attendance_data[student][record_date] = record.status

    # Handle search functionality if search is passed in the URL
    search = request.GET.get('search', '').strip()
    if search:
        attendance_list = [
            {
                "student": student,
                "stack": student.stack.stack_name if student.stack else 'N/A',
                "dates": [
                    "H" if day.weekday() in [5, 6] else attendance_data[student].get(day, "--")
                    for day in last_30_days
                ],
            }
            for student in attendance_data if search.lower() in f'{student.first_name} {student.last_name}'.lower()
        ]
    else:
        attendance_list = [
            {
                "student": student,
                "stack": student.stack.stack_name if student.stack else 'N/A',
                "dates": [
                    "H" if day.weekday() in [5, 6] else attendance_data[student].get(day, "--")
                    for day in last_30_days
                ],
            }
            for student in attendance_data
        ]

    # Check if download is requested
    if request.GET.get('download') == 'excel':
        return generate_excel(attendance_list, last_30_days, search)

    return render(request, "sheet.html", {"attendance_list": attendance_list, "all_dates": last_30_days, "search": search})


def generate_excel(attendance_list, last_30_days, search):
    # Create a Pandas DataFrame
    data = []
    for idx, attendance in enumerate(attendance_list, 1):
        row = [
            idx,
            f'{attendance["student"].first_name} {attendance["student"].last_name}',
            attendance["stack"]
        ] + attendance["dates"]
        data.append(row)

    # Define column headers
    columns = ['No.', 'Student', 'Stack'] + [str(date) for date in last_30_days]

    df = pd.DataFrame(data, columns=columns)

    # Create an HTTP response to download the file as an Excel
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = f'attachment; filename="attendance_sheet_{search if search else "all_students"}.xlsx"'
    
    # Write the DataFrame to the response as an Excel file
    with pd.ExcelWriter(response, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)

    return response
