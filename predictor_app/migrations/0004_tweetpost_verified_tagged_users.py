from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predictor_app', '0003_tweetpost_ridge_features'),
    ]

    operations = [
        migrations.AddField(
            model_name='tweetpost',
            name='tagged_users_count',
            field=models.IntegerField(default=0, help_text='Number of tagged/mentioned users'),
        ),
        migrations.AddField(
            model_name='tweetpost',
            name='is_verified',
            field=models.BooleanField(default=False, help_text='Is the author verified'),
        ),
    ]
