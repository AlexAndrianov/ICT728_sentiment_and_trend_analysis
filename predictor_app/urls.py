from django.urls import path
from django.views.generic import TemplateView

from . import views

urlpatterns = [
    path("", views.login, name="login"),
    path("selector/", views.landing, name="landing"),
    path("dashboard/", views.index, name="index"),
    path("trends/", views.trends, name="trends"),
    path("trends-cloud/", TemplateView.as_view(template_name="predictor_app/trends_cloud.html"), name="trends_cloud"),
    path("trend-cloud-data/", views.get_trends_cloud_data, name="get_trends_cloud_data"),
    path("tweets/", views.tweet_list, name="tweet_list"),
    path("save-to-dashboard/", views.save_to_dashboard, name="save_to_dashboard"),
    path("remove-from-dashboard/", views.remove_from_dashboard, name="remove_from_dashboard"),
    path("forecast-views/<int:tweet_id>/", views.forecast_views, name="forecast_views"),
    path("sentiment/<int:tweet_id>/", views.sentiment, name="sentiment"),
    path("trends-iteration/", views.get_trends_iteration, name="get_trends_iteration"),
    path("trend-analytics/", views.get_trend_analytics, name="get_trend_analytics"),
]