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

    var duration = selectedTime * 60000;

    document.getElementById("videoStream").src = "{% url 'video_stream' %}";

    setTimeout(function() {
      document.getElementById('videoSection').style.display = 'none';
    }, duration);
  });
</script>

{% endblock %}






def gen(request):
    cap = cv2.VideoCapture(0)
    duration = request.session.get('duration', 0)

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed_time = time.time() - start_time
        if elapsed_time >= duration:
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
