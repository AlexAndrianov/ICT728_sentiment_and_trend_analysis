from django.urls import path

from . import views

urlpatterns = [
    path("", views.landing, name="landing"),
    path("dashboard/", views.index, name="index"),
    path("tweets/", views.tweet_list, name="tweet_list"),
    path("save-to-dashboard/", views.save_to_dashboard, name="save_to_dashboard"),
    path("remove-from-dashboard/", views.remove_from_dashboard, name="remove_from_dashboard"),
    path("forecast-views/<int:tweet_id>/", views.forecast_views, name="forecast_views"),
    path("sentiment/<int:tweet_id>/", views.sentiment, name="sentiment"),
]