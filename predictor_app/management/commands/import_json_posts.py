import json
from datetime import datetime

from django.core.management.base import BaseCommand
from django.utils.dateparse import parse_datetime

from predictor_app.models import TweetPost


class Command(BaseCommand):
    help = "Import TweetPost records from a JSON file into the DB"

    def add_arguments(self, parser):
        parser.add_argument(
            "--json",
            type=str,
            default="ml_models_design/posts/final_edited_json_meaningful_hashtags_translated.json",
            help="Path to the JSON file (relative to project root)",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=50000,
            help="Bulk insert batch size",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=0,
            help="Optional max number of items to process (0 = no limit)",
        )

    def handle(self, *args, **options):
        json_path: str = options["json"]
        batch_size: int = int(options["batch_size"])
        limit: int = int(options["limit"])

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise TypeError(f"Expected top-level JSON array, got {type(data)}")

        created_total = 0
        processed_total = 0

        batch: list[TweetPost] = []

        def _parse_created_at(v):
            if v in (None, ""):
                return None
            if isinstance(v, datetime):
                return v
            if isinstance(v, str):
                dt = parse_datetime(v)
                if dt is not None:
                    return dt
                try:
                    return datetime.fromisoformat(v.replace("Z", "+00:00"))
                except Exception:
                    return None
            return None

        def _parse_real_views(v):
            if v in (None, "", "null"):
                return None
            try:
                return int(float(v))
            except Exception:
                return None

        for item in data:
            if limit and processed_total >= limit:
                break
            processed_total += 1

            if not isinstance(item, dict):
                continue

            tweet_id = (item.get("tweet_id") or "").strip()
            if not tweet_id:
                continue

            created_at = _parse_created_at(item.get("created_at"))
            if created_at is None:
                continue

            posted_hour = item.get("posted_hour")
            if posted_hour in (None, ""):
                posted_hour = int(created_at.astimezone().hour)

            obj = TweetPost(
                tweet_id=tweet_id,
                username=(item.get("username") or "Unknown"),
                screen_name=(item.get("screen_name") or "unknown"),
                content=(item.get("content") or ""),
                created_at=created_at,
                retweet_count=int(item.get("retweet_count") or 0),
                like_count=int(item.get("like_count") or 0),
                profile_image_url=(item.get("profile_image_url") or ""),
                posted_hour=int(posted_hour or 0),
                photos_count=int(item.get("photos_count") or 0),
                videos_count=int(item.get("videos_count") or 0),
                hashtags_count=int(item.get("hashtags_count") or 0),
                tagged_users_count=int(item.get("tagged_users_count") or 0),
                followers_count=int(item.get("followers_count") or 0),
                posts_count=int(item.get("posts_count") or 0),
                following_count=int(item.get("following_count") or 0),
                is_verified=bool(item.get("is_verified") or False),
                description_error_count=int(item.get("description_error_count") or 0),
                swear_word_count=int(item.get("swear_word_count") or 0),
                word_count=int(item.get("word_count") or 0),
                emoji_count=int(item.get("emoji_count") or 0),
                uppercase_word_count=int(item.get("uppercase_word_count") or 0),
                real_views=_parse_real_views(item.get("real_views")),
                hashtags=(item.get("hashtags") or ""),
                real_sentiment=(item.get("real_sentiment") or "neutral"),
                is_dataset_tweet=bool(item.get("is_dataset_tweet") or False),
            )

            batch.append(obj)
            if len(batch) >= batch_size:
                res = TweetPost.objects.bulk_create(batch, ignore_conflicts=True)
                created_total += len(res)
                batch.clear()

        if batch:
            res = TweetPost.objects.bulk_create(batch, ignore_conflicts=True)
            created_total += len(res)

        self.stdout.write(
            self.style.SUCCESS(
                f"Processed {processed_total} JSON items. Inserted {created_total} new TweetPost rows."
            )
        )
