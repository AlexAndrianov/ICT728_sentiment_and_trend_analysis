from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("predictor_app", "0006_sentiment_model_settings"),
    ]

    operations = [
        migrations.AddField(
            model_name="tweetpost",
            name="real_views",
            field=models.BigIntegerField(blank=True, help_text="Real (observed) views from dataset", null=True),
        ),
        migrations.AddField(
            model_name="tweetpost",
            name="real_sentiment",
            field=models.CharField(
                choices=[("negative", "Negative"), ("neutral", "Neutral"), ("positive", "Positive")],
                default="neutral",
                help_text="Real (observed) sentiment label from dataset",
                max_length=16,
            ),
        ),
        migrations.AddField(
            model_name="tweetpost",
            name="is_dataset_tweet",
            field=models.BooleanField(default=False, help_text="Imported from CSV dataset"),
        ),
    ]
