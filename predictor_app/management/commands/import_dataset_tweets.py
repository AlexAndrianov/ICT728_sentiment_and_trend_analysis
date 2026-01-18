import csv
from datetime import datetime
from django.core.management.base import BaseCommand

from predictor_app.models import TweetPost
from predictor_app.management.commands.fetch_tweets import (
    count_emojis,
    count_swear_words,
    count_uppercase_words,
    count_words,
)


_FIXED_TWEET_SENTIMENTS = {
    "1779560388859760867": "positive",
}


class Command(BaseCommand):
    help = "Import a small sample of tweets from ml_models_design/Twitter- datasets.csv into the DB"

    def add_arguments(self, parser):
        parser.add_argument(
            "--csv",
            type=str,
            default="ml_models_design/Twitter- datasets.csv",
            help="Path to the dataset CSV (relative to project root)",
        )

    def handle(self, *args, **options):
        csv_path = options["csv"]

        created = 0
        skipped = 0

        def _create_from_row(row: dict, *, real_sentiment: str) -> bool:
            text = (row.get("description") or "").strip()
            tweet_id = (row.get("id") or "").strip().strip('"')
            if not tweet_id:
                return False

            if TweetPost.objects.filter(tweet_id=tweet_id).exists():
                return False

            name = (row.get("name") or "Unknown").strip().strip('"')
            screen_name = (row.get("user_posted") or "unknown").strip().strip('"')

            date_raw = (row.get("date_posted") or "").strip()
            date_raw = date_raw.strip('"')
            created_at = None
            if date_raw:
                try:
                    created_at = datetime.fromisoformat(date_raw.replace("Z", "+00:00"))
                except Exception:
                    created_at = None
            if created_at is None:
                created_at = datetime.utcnow().astimezone()

            try:
                views = row.get("views")
                real_views = int(float(views)) if views not in (None, "", "null") else None
            except Exception:
                real_views = None

            TweetPost.objects.create(
                tweet_id=tweet_id,
                username=name or "Unknown",
                screen_name=screen_name or "unknown",
                content=text,
                created_at=created_at,
                retweet_count=int(row.get("reposts") or 0),
                like_count=int(row.get("likes") or 0),
                profile_image_url=(row.get("profile_image_link") or "").strip().strip('"'),
                posted_hour=int(created_at.astimezone().hour),
                photos_count=0,
                videos_count=0,
                hashtags_count=0,
                tagged_users_count=0,
                followers_count=int(row.get("followers") or 0),
                posts_count=int(row.get("posts_count") or 0),
                following_count=int(row.get("following") or 0),
                is_verified=str(row.get("is_verified") or "").lower() == "true",
                description_error_count=0,
                swear_word_count=count_swear_words(text),
                word_count=count_words(text),
                emoji_count=count_emojis(text),
                uppercase_word_count=count_uppercase_words(text),
                real_views=real_views,
                real_sentiment=real_sentiment,
                is_dataset_tweet=True,
            )

            return True
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            fixed_remaining = dict(_FIXED_TWEET_SENTIMENTS)

            for row in reader:
                if not fixed_remaining:
                    break

                tweet_id = (row.get("id") or "").strip().strip('"')
                if not tweet_id:
                    continue

                real_sentiment = fixed_remaining.get(tweet_id)
                if not real_sentiment:
                    continue

                if _create_from_row(row, real_sentiment=real_sentiment):
                    created += 1
                else:
                    skipped += 1

                fixed_remaining.pop(tweet_id, None)

        self.stdout.write(self.style.SUCCESS(f"Imported {created} dataset tweets. Skipped {skipped}."))
