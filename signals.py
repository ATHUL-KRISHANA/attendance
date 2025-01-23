from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Student, AttendanceRecord
import datetime

@receiver(post_save, sender=Student)
def create_or_update_attendance(sender, instance, created, **kwargs):
    if created:
        AttendanceRecord.objects.create(
            student=instance,
            date=datetime.date.today(),
            status=instance.present
        )
    else:
        today = datetime.date.today()
        attendance_record, _ = AttendanceRecord.objects.get_or_create(
            student=instance,
            date=today,
            defaults={'status': instance.present}
        )
        if attendance_record.status != instance.present:
            attendance_record.status = instance.present
            attendance_record.save()
