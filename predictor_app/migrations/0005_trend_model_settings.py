from django.db import migrations, models


def create_default_settings(apps, schema_editor):
    TrendModelSettings = apps.get_model("predictor_app", "TrendModelSettings")
    TrendModelSettings.objects.get_or_create(id=1, defaults={"selected_model": "ridge"})


class Migration(migrations.Migration):

    dependencies = [
        ("predictor_app", "0004_tweetpost_verified_tagged_users"),
    ]

    operations = [
        migrations.CreateModel(
            name="TrendModelSettings",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                (
                    "selected_model",
                    models.CharField(
                        choices=[
                            ("ridge", "Ridge (Sentence-BERT + numeric)"),
                            ("xgb", "XGBoost (GridSearch best)"),
                        ],
                        default="ridge",
                        max_length=64,
                    ),
                ),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.RunPython(create_default_settings, migrations.RunPython.noop),
    ]
