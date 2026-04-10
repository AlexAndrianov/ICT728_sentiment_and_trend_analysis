
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

SWEAR_WORDS = {
    "damn", "hell", "shit", "fuck", "fucking", "bitch", "crap", "ass", "bastard",
    "dick", "piss", "bloody", "slut", "whore", "idiot", "moron", "stupid"
}

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)

HASHTAG_PATTERN = re.compile(r"(?<!\w)#([A-Za-z0-9_]+)")
MENTION_PATTERN = re.compile(r"(?<!\w)@([A-Za-z0-9_.]+)")
WORD_PATTERN = re.compile(r"\b\w+\b")


def safe_int(value: Any, default: int = 0) -> int:
    if pd.isna(value):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_iso_datetime(value: Any) -> Optional[str]:
    raw = clean_text(value).strip('"').strip("'")
    if not raw:
        return None

    for candidate in [raw, raw.replace("Z", "+00:00")]:
        try:
            dt = datetime.fromisoformat(candidate)
            return dt.isoformat()
        except ValueError:
            continue

    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.isoformat()
        except ValueError:
            continue
    return None


def extract_hashtags(text: str) -> List[str]:
    return list(dict.fromkeys(HASHTAG_PATTERN.findall(text)))


def extract_mentions(text: str) -> List[str]:
    return list(dict.fromkeys(MENTION_PATTERN.findall(text)))


def count_emojis(text: str) -> int:
    return sum(len(match) for match in EMOJI_PATTERN.findall(text))


def count_swear_words(text: str) -> int:
    words = [word.lower() for word in WORD_PATTERN.findall(text)]
    return sum(1 for word in words if word in SWEAR_WORDS)


def count_uppercase_words(text: str) -> int:
    words = WORD_PATTERN.findall(text)
    return sum(1 for word in words if len(word) > 1 and word.isupper())


def build_common_record(
    *,
    tweet_id: str,
    username: str,
    screen_name: str,
    content: str,
    created_at: Optional[str],
    like_count: int = 0,
    retweet_count: int = 0,
    profile_image_url: str = "",
    photos_count: int = 0,
    videos_count: int = 0,
    followers_count: int = 0,
    posts_count: int = 0,
    following_count: int = 0,
    is_verified: bool = False,
    real_views: Optional[int] = None,
    real_sentiment: str = "neutral",
) -> Dict[str, Any]:
    hashtags = extract_hashtags(content)
    mentions = extract_mentions(content)
    word_count = len(WORD_PATTERN.findall(content))
    posted_hour = 0

    if created_at:
        try:
            posted_hour = datetime.fromisoformat(created_at.replace("Z", "+00:00")).hour
        except ValueError:
            posted_hour = 0

    return {
        "tweet_id": str(tweet_id),
        "username": username or "unknown_user",
        "screen_name": screen_name or "unknown_user",
        "content": content,
        "created_at": created_at,
        "retweet_count": retweet_count,
        "like_count": like_count,
        "profile_image_url": profile_image_url,
        "posted_hour": posted_hour,
        "photos_count": photos_count,
        "videos_count": videos_count,
        "hashtags_count": len(hashtags),
        "tagged_users_count": len(mentions),
        "followers_count": followers_count,
        "posts_count": posts_count,
        "following_count": following_count,
        "is_verified": bool(is_verified),
        "description_error_count": 0,
        "swear_word_count": count_swear_words(content),
        "word_count": word_count,
        "emoji_count": count_emojis(content),
        "uppercase_word_count": count_uppercase_words(content),
        "real_views": real_views,
        "hashtags": ",".join(hashtags),
        "real_sentiment": real_sentiment,
        "is_dataset_tweet": True,
    }


def convert_facebook_dataset(csv_path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        content = clean_text(row.get("comment_text"))
        if not content:
            continue

        created_at = clean_iso_datetime(row.get("date_created"))
        attached_files = clean_text(row.get("attached_files"))
        video_length = row.get("video_length")

        photos_count = 1 if attached_files else 0
        videos_count = 1 if not pd.isna(video_length) else 0

        record = build_common_record(
            tweet_id=f"facebook_{clean_text(row.get('comment_id')) or clean_text(row.get('post_id'))}",
            username=clean_text(row.get("user_name")),
            screen_name=clean_text(row.get("user_name")).replace(" ", "_").lower(),
            content=content,
            created_at=created_at,
            like_count=safe_int(row.get("num_likes")),
            retweet_count=0,
            profile_image_url="",
            photos_count=photos_count,
            videos_count=videos_count,
            followers_count=0,
            posts_count=0,
            following_count=0,
            is_verified=False,
            real_views=None,
            real_sentiment="neutral",
        )
        records.append(record)

    return records


def convert_instagram_dataset(csv_path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        content = clean_text(row.get("comment"))
        if not content:
            continue

        created_at = clean_iso_datetime(row.get("comment_date"))
        existing_hashtag_field = clean_text(row.get("hashtag_comment"))
        existing_tags = clean_text(row.get("tagged_users_in_comment"))

        record = build_common_record(
            tweet_id=f"instagram_{clean_text(row.get('comment_id')) or clean_text(row.get('post_id'))}",
            username=clean_text(row.get("comment_user")),
            screen_name=clean_text(row.get("comment_user")).replace(" ", "_").lower(),
            content=content,
            created_at=created_at,
            like_count=safe_int(row.get("likes_number")),
            retweet_count=0,
            profile_image_url="",
            photos_count=1,
            videos_count=0,
            followers_count=0,
            posts_count=0,
            following_count=0,
            is_verified=False,
            real_views=None,
            real_sentiment="neutral",
        )

        hashtags = extract_hashtags(content)
        if existing_hashtag_field:
            hashtags.extend([tag.strip().lstrip("#") for tag in existing_hashtag_field.split(",") if tag.strip()])
            hashtags = list(dict.fromkeys(hashtags))
            record["hashtags"] = ",".join(hashtags)
            record["hashtags_count"] = len(hashtags)

        if existing_tags:
            extra_mentions = [tag.strip().lstrip("@") for tag in existing_tags.split(",") if tag.strip()]
            base_mentions = extract_mentions(content)
            merged_mentions = list(dict.fromkeys(base_mentions + extra_mentions))
            record["tagged_users_count"] = len(merged_mentions)

        records.append(record)

    return records


def convert_tiktok_dataset(csv_path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        content = clean_text(row.get("comment_text"))
        if not content:
            continue

        created_at = clean_iso_datetime(row.get("date_created"))

        record = build_common_record(
            tweet_id=f"tiktok_{clean_text(row.get('comment_id')) or clean_text(row.get('post_id'))}",
            username=clean_text(row.get("commenter_user_name")),
            screen_name=clean_text(row.get("commenter_user_name")).replace(" ", "_").lower(),
            content=content,
            created_at=created_at,
            like_count=safe_int(row.get("num_likes")),
            retweet_count=0,
            profile_image_url="",
            photos_count=0,
            videos_count=1,
            followers_count=0,
            posts_count=0,
            following_count=0,
            is_verified=False,
            real_views=None,
            real_sentiment="neutral",
        )
        records.append(record)

    return records


def convert_all_datasets_to_single_json(
    facebook_csv: str,
    instagram_csv: str,
    tiktok_csv: str,
    output_json_path: str = "converted_social_to_twitter_format.json",
) -> List[Dict[str, Any]]:
    facebook_records = convert_facebook_dataset(facebook_csv)
    instagram_records = convert_instagram_dataset(instagram_csv)
    tiktok_records = convert_tiktok_dataset(tiktok_csv)

    all_records = facebook_records + instagram_records + tiktok_records

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    return all_records

if __name__ == "__main__":
    facebook_csv = "/content/Facebook-datasets.csv"
    instagram_csv = "/content/Instagram-datasets.csv"
    tiktok_csv = "/content/TikTok-datasets.csv"
    output_json = "/content/converted_social_to_twitter_format.json"

    records = convert_all_datasets_to_single_json(
        facebook_csv=facebook_csv,
        instagram_csv=instagram_csv,
        tiktok_csv=tiktok_csv,
        output_json_path=output_json,
    )

    print(f"Total converted records: {len(records)}")
    print("First 2 records:")
    print(json.dumps(records[:2], indent=2, ensure_ascii=False))
