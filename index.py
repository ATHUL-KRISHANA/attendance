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
