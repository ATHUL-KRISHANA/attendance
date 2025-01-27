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
from django.http import StreamingHttpResponse




face = face_analysis()
embedder = FaceNet()

data_dir = 'media'
pickle_path = 'enrollment'
pickle_path2 = 'enrollment2'


def recognize_face(face_embedding,dir_path):
        min_distance = float('inf')
        recognized_person = "UNKNOWN"
        recognition_accuracy = 0

        for file in os.listdir(dir_path):
            if file.endswith('_embeddings.pkl'):
                stored_embeddings = pickle.load(open(os.path.join(dir_path, file), 'rb'))
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
        first_name = data.get("first_name").strip()
        last_name = data.get("last_name").strip()
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

# attendance --------------------------------------------------------


def gen(request):
    cap = cv2.VideoCapture(1)
    
    # Get the duration from the GET parameters passed in the URL
    duration = int(request.GET.get('duration', 0))  # in milliseconds from JavaScript

    # Convert milliseconds to seconds for time calculations
    duration_seconds = duration / 1000

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        if elapsed_time >= duration_seconds:
            break  

        _, boxes, _ = face.face_detection(frame_arr=frame, frame_status=True, model='tiny')

        if len(boxes) > 0:
            for box in boxes:
                x, y, w, h = box
                face_crop = frame[y:y + w, x:x + h]

                if face_crop is not None and face_crop.size > 0:
                    face_resized = cv2.resize(face_crop, (160, 160))

                    face_embedding = embedder.embeddings([face_resized])[0]
                    person, accuracy = recognize_face(face_embedding, pickle_path)

                    if person != "UNKNOWN" and accuracy > 72:
                        name_parts = person.split()
                        first_name = name_parts[0]
                        last_name = name_parts[1] if len(name_parts) > 1 else ""

                        try:
                            student = get_object_or_404(Student, first_name=first_name, last_name=last_name)
                            student.present = "Present"
                            student.save()
                        except Student.DoesNotExist:
                            pass

                        cv2.putText(frame, f"{person} ({accuracy:.2f}%)", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.rectangle(frame, (x, y), (x + h, y + w), (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cap.release()

def video_stream(request):
    return StreamingHttpResponse(gen(request), content_type='multipart/x-mixed-replace; boundary=frame')


def attendance(request):
    if request.method == "POST":
        selected_time = int(request.POST.get("time", 1))
        duration = selected_time * 60  
        
        return render(request, 'index.html', {'duration': duration})

    return render(request, 'index.html')

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
                    
                    person, accuracy = recognize_face(face_embedding,pickle_path2)
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
                                cap.release()
                                cv2.destroyAllWindows()                             
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
    # Get query parameters
    selected_month = request.GET.get('month', datetime.now().strftime('%Y-%m'))  # Default to current month
    search = request.GET.get('search', '')
    download = request.GET.get('download', '')

    # Parse the selected month and calculate start and end dates
    try:
        start_date = datetime.strptime(selected_month, "%Y-%m")
    except ValueError:
        start_date = datetime.now().replace(day=1)
    end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    
    # Filter attendance records for the selected month
    attendance_records = AttendanceRecord.objects.filter(
        date__range=(start_date, end_date)
    )
    if search:
        attendance_records = attendance_records.filter(student__first_name__icontains=search)

    # Collect all unique dates for the month
    all_dates = attendance_records.values_list('date', flat=True).distinct()
    all_dates = sorted(set(all_dates))

    # Process attendance data
    attendance_list = []
    for student in attendance_records.values('student').distinct():
        student_attendance = attendance_records.filter(student=student['student'])
        statuses = []
        total_present = 0
        total_absent = 0
        for date in all_dates:
            record = student_attendance.filter(date=date).first()
            if record:
                statuses.append(record.status)
                if record.status == "Present":
                    total_present += 1
                elif record.status == "Absent":
                    total_absent += 1
            else:
                statuses.append("N/A")
        
        attendance_list.append({
            "student": student_attendance.first().student,
            "stack": student_attendance.first().student.stack.stack_name,
            "dates": statuses,
            "total_present": total_present,
            "total_absent": total_absent,
        })

    # Generate Excel if download is requested
    if download == 'excel':
        return generate_excel(attendance_list, all_dates, search, selected_month)

    # Get available months
    available_months = AttendanceRecord.objects.dates('date', 'month')

    context = {
        "attendance_list": attendance_list,
        "all_dates": all_dates,
        "available_months": available_months,
        "selected_month": selected_month,
        "search": search,
    }
    return render(request, "sheet.html", context)


def generate_excel(attendance_list, all_dates, search, selected_month):
    # Create Excel data
    data = []
    headers = ['No.', 'Student', 'Stack'] + [date.strftime('%d-%m-%Y') for date in all_dates] + ['Total Present', 'Total Absent']

    for idx, attendance in enumerate(attendance_list, start=1):
        row = [idx, f"{attendance['student'].first_name} {attendance['student'].last_name}", attendance['stack']]
        row.extend(attendance['dates'])
        row.append(attendance['total_present'])
        row.append(attendance['total_absent'])
        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data, columns=headers)

    # Create Excel file
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = f'attachment; filename=attendance_{selected_month}.xlsx'
    with pd.ExcelWriter(response, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name="Attendance")

    return response







{% extends 'base.html' %}

{% block content %}
<div class="container my-5 p-5 shadow-lg" style="border-radius:10px;">

    <div class="d-flex justify-content-between align-items-center">
        <h2>Attendance Sheet</h2>
        <a href="{% url 'sheet' %}?download=excel&month={{ selected_month }}{% if search %}&search={{ search }}{% endif %}" class="btn btn-success mt-4">Download</a>
    </div>
    
    <form method="GET" action="{% url 'sheet' %}" class="mt-3">
        <div class="input-group mb-3">
            <label for="month-select" class="form-label me-3">Select Month:</label>
            <select id="month-select" name="month" class="form-select">
                {% for month_date in available_months %}
                <option value="{{ month_date|date:'Y-m' }}" {% if month_date|date:'Y-m' == selected_month %}selected{% endif %}>
                    {{ month_date|date:'F Y' }}
                </option>
                {% endfor %}
            </select>
            <input type="text" name="search" class="form-control ms-3" placeholder="Search by Student Name" value="{% if search %}{{ search }}{% endif %}">
            <button class="btn btn-outline-primary ms-3" type="submit">Filter</button>
        </div>
    </form>

    {% if attendance_list %}
    <div class="table-responsive">
        <table class="table table-bordered">
            <thead>
                <tr>
                    <th>No.</th>
                    <th>Student</th>
                    <th>Stack</th>
                    {% for date in all_dates %}
                    <th>{{ date|date:"d M" }}</th>
                    {% endfor %}
                    <th>Total Present</th>
                    <th>Total Absent</th>
                </tr>
            </thead>
            <tbody>
                {% for attendance in attendance_list %}
                    <tr>
                        <td>{{ forloop.counter }}</td>
                        <td>{{ attendance.student.first_name }} {{ attendance.student.last_name }}</td>
                        <td>{{ attendance.stack }}</td>
                        {% for status in attendance.dates %}
                        <td class="{% if status == 'Absent' %}text-danger{% elif status == 'Present' %}text-success{% endif %}">
                            {{ status }}
                        </td>
                        {% endfor %}
                        <td>{{ attendance.total_present }}</td>
                        <td>{{ attendance.total_absent }}</td>
                    </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% else %}
    <p class="text-center mt-4">No attendance data available for the selected month or search query.</p>
    {% endif %}
</div>
{% endblock %}
_________________________________________

{% extends 'base.html' %}

{% block content %}
<div class="container mt-4">

  <div id="videoSection" class="mt-4 text-center " style="display: none;">
    <div class="d-flex justify-content-center">
      <img id="videoStream" src="" class="img-fluid shadow-lg rounded mb-4" alt="Video Stream">
    </div>
  </div>

  <div class="row mt-4 justify-content-center mb-5">
    <div class="col-md-4 d-flex justify-content-center">
      <div class="box p-4 shadow rounded bg-white">
        <h4 class="mb-4 text-center">Set Time</h4>
        <form id="timeForm" method="POST" action="{% url 'attendance' %}">
          {% csrf_token %}
          <div class="mb-3">
            <select class="form-select" id="timeDropdown" name="time">
              <option value="1">1 Minute</option>
              <option value="30">30 Minutes</option>
              <option value="60">60 Minutes</option>
            </select>
          </div>
          <button class="btn btn-primary w-100" id="startButton" type="submit">Start</button>
        </form>
      </div>
    </div>
  </div>



</div>

<script>
  document.getElementById('timeForm').addEventListener('submit', function(event) {
    event.preventDefault(); 
  
    document.getElementById('videoSection').style.display = 'block';
  
    var selectedTime = document.getElementById("timeDropdown").value;
    var duration = selectedTime * 60000; // Convert minutes to milliseconds
  
    // Pass the duration to the video stream
    document.getElementById("videoStream").src = "{% url 'video_stream' %}?duration=" + duration;
  
    setTimeout(function() {
      document.getElementById('videoSection').style.display = 'none';
    }, duration);
  });
  
  
</script>

{% endblock %}
