from django.contrib import admin
from django.utils.html import format_html
from datetime import date
from .models import Student,AttendanceRecord,Stack,Faculty

from django.utils.timezone import now

class StudentAdmin(admin.ModelAdmin):
    list_display = ['full_name', 'stack', 'phone', 'email', 'display_status', 'highlight_long_tenure']
    list_filter = ['batch', 'present']
    actions = ['set_absent', 'set_interview', 'set_holiday', 'set_present']
    search_fields = ['first_name', 'last_name']

    def display_status(self, obj):
        if obj.present == 'Present':
            return "🟢 Present"
        elif obj.present == 'Absent':
            return "🔴 Absent"
        elif obj.present == 'Interview':
            return "🔵 Interview"
        elif obj.present == "Holiday":
            return "🟡 Holiday"
        return "Unknown"
    
    display_status.short_description = "Status"

    def full_name(self, obj):
        return f"{obj.first_name} {obj.last_name}"
    
    full_name.short_description = "Full Name"

    #------------------------------------------------------------------------------

    def set_absent(self, request, queryset):
        queryset.update(present='Absent')        
        for student in queryset:
            AttendanceRecord.objects.update_or_create(
                student=student, 
                date=now().date(),  
                defaults={'status': 'Absent'}
            )
        
        self.message_user(request, "Selected students have been set to 'Absent'.")
    
    set_absent.short_description = "Set selected students to Absent"
    #------------------------------------------------------------------------------

    def set_interview(self, request, queryset):
        queryset.update(present="Interview")
        for student in queryset:
            AttendanceRecord.objects.update_or_create(
                student=student, 
                date=now().date(),
                defaults={'status': 'Interview'}
            )
        self.message_user(request, "Selected students have been set to 'Interview'.")
    
    set_interview.short_description = "Set selected students to Interview"

    #-----------------------------------------------------------------------------

    def set_holiday(self, request, queryset):
        queryset.update(present="Holiday")
        for student in queryset:
            AttendanceRecord.objects.update_or_create(
                student=student, 
                date=now().date(),
                defaults={'status': 'Holiday'}
            )
        self.message_user(request, "Selected students have been set to 'Holiday'.")
    
    set_holiday.short_description = "Set selected students to Holiday"

    #-----------------------------------------------------------------------------    

    def set_present(self, request, queryset):
        queryset.update(present="Present")
        for student in queryset:
            AttendanceRecord.objects.update_or_create(
                student=student, 
                date=now().date(),
                defaults={'status': 'Present'}
            )
        self.message_user(request, "Selected students have been set to 'Present'.")
    
    set_present.short_description = "Set selected students to Present"

    #-----------------------------------------------------------------------------

    def highlight_long_tenure(self, obj):
        if obj.join_date and (date.today() - obj.join_date).days > 90:
            formatted_date = obj.join_date.strftime('%b. %d, %Y')
            return format_html(
                '<span style="background-color: yellow; color: black;">{}</span>',
                formatted_date
            )
        return obj.join_date.strftime('%b. %d, %Y') if obj.join_date else "Unknown"

    highlight_long_tenure.short_description = "Join Date"


admin.site.register(Student, StudentAdmin)
admin.site.register(AttendanceRecord)
admin.site.register(Stack)
admin.site.register(Faculty)
