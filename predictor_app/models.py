from django.db import models
from django.utils import timezone


class TweetPost(models.Model):
    """Model for storing Twitter posts"""
    tweet_id = models.CharField(max_length=50, unique=True, help_text="Unique tweet ID")
    username = models.CharField(max_length=100, help_text="User display name")
    screen_name = models.CharField(max_length=100, help_text="User screen name")
    content = models.TextField(help_text="Tweet content")
    created_at = models.DateTimeField(help_text="Tweet creation date")
    retweet_count = models.IntegerField(default=0, help_text="Number of retweets")
    like_count = models.IntegerField(default=0, help_text="Number of likes")
    profile_image_url = models.URLField(blank=True, help_text="User profile image URL")

    posted_hour = models.IntegerField(default=0, help_text="Local hour of posting (0-23)")
    photos_count = models.IntegerField(default=0, help_text="Number of photos attached")
    videos_count = models.IntegerField(default=0, help_text="Number of videos attached")
    hashtags_count = models.IntegerField(default=0, help_text="Number of hashtags")
    tagged_users_count = models.IntegerField(default=0, help_text="Number of tagged/mentioned users")
    followers_count = models.IntegerField(default=0, help_text="Author followers count")
    posts_count = models.IntegerField(default=0, help_text="Author total posts count")
    following_count = models.IntegerField(default=0, help_text="Author following count")
    is_verified = models.BooleanField(default=False, help_text="Is the author verified")

    description_error_count = models.IntegerField(default=0, help_text="Grammar error count (optional/offline)")
    swear_word_count = models.IntegerField(default=0, help_text="Swear words count (heuristic)")
    word_count = models.IntegerField(default=0, help_text="Word count")
    emoji_count = models.IntegerField(default=0, help_text="Emoji count")
    uppercase_word_count = models.IntegerField(default=0, help_text="Uppercase word count")

    real_views = models.BigIntegerField(null=True, blank=True, help_text="Real (observed) views from dataset")
    real_sentiment = models.CharField(
        max_length=16,
        default="neutral",
        choices=[
            ("negative", "Negative"),
            ("neutral", "Neutral"),
            ("positive", "Positive"),
        ],
        help_text="Real (observed) sentiment label from dataset",
    )
    is_dataset_tweet = models.BooleanField(default=False, help_text="Imported from CSV dataset")

    updated_at = models.DateTimeField(auto_now=True, help_text="Record update date")

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Tweet"
        verbose_name_plural = "Tweets"

    def __str__(self):
        return f"@{self.screen_name}: {self.content[:50]}..."


class SavedPost(models.Model):
    original_tweet = models.ForeignKey(TweetPost, on_delete=models.CASCADE, related_name='saved_posts')
    position_x = models.IntegerField(help_text="X position in grid")
    position_y = models.IntegerField(help_text="Y position in grid")
    saved_at = models.DateTimeField(auto_now_add=True, help_text="When post was saved")

    class Meta:
        ordering = ['position_y', 'position_x']
        unique_together = ['position_x', 'position_y']
        verbose_name = "Saved Post"
        verbose_name_plural = "Saved Posts"

    def __str__(self):
        return f"Saved at ({self.position_x}, {self.position_y}): {self.original_tweet.screen_name}"


class TrendModelSettings(models.Model):
    selected_model = models.CharField(
        max_length=64,
        default="ridge",
        choices=[
            ("ridge", "Ridge (Sentence-BERT + numeric)"),
            ("xgb", "XGBoost (GridSearch best)"),
        ],
    )
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Trend model: {self.selected_model}"


class SentimentModelSettings(models.Model):
    selected_model_id = models.IntegerField(
        default=1,
        choices=[
            (1, "Naive_Bayes_Model_With_Simple_Tokenizer"),
            (2, "Stacking_Classifier_Logistic_Regression_Plus_SVC"),
            (3, "PyTorch_Neural_Network_Model"),
        ],
    )
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Sentiment model: {self.selected_model_id}"
