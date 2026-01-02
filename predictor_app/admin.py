from django.contrib import admin

from .models import SentimentModelSettings, TrendModelSettings


@admin.register(TrendModelSettings)
class TrendModelSettingsAdmin(admin.ModelAdmin):
    list_display = ("id", "selected_model", "updated_at")

    def has_add_permission(self, request):
        return not TrendModelSettings.objects.exists()

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(SentimentModelSettings)
class SentimentModelSettingsAdmin(admin.ModelAdmin):
    list_display = ("id", "selected_model_id", "updated_at")

    def has_add_permission(self, request):
        return not SentimentModelSettings.objects.exists()

    def has_delete_permission(self, request, obj=None):
        return False
