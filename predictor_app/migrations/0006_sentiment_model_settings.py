from django.db import migrations, models


def create_default_settings(apps, schema_editor):
    SentimentModelSettings = apps.get_model("predictor_app", "SentimentModelSettings")
    SentimentModelSettings.objects.get_or_create(id=1, defaults={"selected_model_id": 1})


class Migration(migrations.Migration):

    dependencies = [
        ("predictor_app", "0005_trend_model_settings"),
    ]

    operations = [
        migrations.CreateModel(
            name="SentimentModelSettings",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                (
                    "selected_model_id",
                    models.IntegerField(
                        choices=[
                            (1, "Naive_Bayes_Model_With_Simple_Tokenizer"),
                            (2, "Stacking_Classifier_Logistic_Regression_Plus_SVC"),
                            (3, "PyTorch_Neural_Network_Model"),
                        ],
                        default=1,
                    ),
                ),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.RunPython(create_default_settings, migrations.RunPython.noop),
    ]
