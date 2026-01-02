from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predictor_app', '0002_savedpost'),
    ]

    operations = [
        migrations.AddField(
            model_name='tweetpost',
            name='posted_hour',
            field=models.IntegerField(default=0, help_text='Local hour of posting (0-23)'),
        ),
        migrations.AddField(
            model_name='tweetpost',
            name='photos_count',
            field=models.IntegerField(default=0, help_text='Number of photos attached'),
        ),
        migrations.AddField(
            model_name='tweetpost',
            name='videos_count',
            field=models.IntegerField(default=0, help_text='Number of videos attached'),
        ),
        migrations.AddField(
            model_name='tweetpost',
            name='hashtags_count',
            field=models.IntegerField(default=0, help_text='Number of hashtags'),
        ),
        migrations.AddField(
            model_name='tweetpost',
            name='followers_count',
            field=models.IntegerField(default=0, help_text='Author followers count'),
        ),
        migrations.AddField(
            model_name='tweetpost',
            name='posts_count',
            field=models.IntegerField(default=0, help_text='Author total posts count'),
        ),
        migrations.AddField(
            model_name='tweetpost',
            name='following_count',
            field=models.IntegerField(default=0, help_text='Author following count'),
        ),
        migrations.AddField(
            model_name='tweetpost',
            name='description_error_count',
            field=models.IntegerField(default=0, help_text='Grammar error count (optional/offline)'),
        ),
        migrations.AddField(
            model_name='tweetpost',
            name='swear_word_count',
            field=models.IntegerField(default=0, help_text='Swear words count (heuristic)'),
        ),
        migrations.AddField(
            model_name='tweetpost',
            name='word_count',
            field=models.IntegerField(default=0, help_text='Word count'),
        ),
        migrations.AddField(
            model_name='tweetpost',
            name='emoji_count',
            field=models.IntegerField(default=0, help_text='Emoji count'),
        ),
        migrations.AddField(
            model_name='tweetpost',
            name='uppercase_word_count',
            field=models.IntegerField(default=0, help_text='Uppercase word count'),
        ),
    ]
