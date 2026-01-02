from django.core.management.base import BaseCommand
from django.conf import settings
import requests
from datetime import datetime
from predictor_app.models import SavedPost, TweetPost
import re
import time
from pathlib import Path
import shutil
import requests.exceptions


EMOJI_PATTERN = re.compile(
    "[\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended
    "\U00002600-\U000026FF"  # misc symbols
    "]+",
    flags=re.UNICODE,
)


def count_words(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))


def count_emojis(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(EMOJI_PATTERN.findall(text))


def count_uppercase_words(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(re.findall(r"\b[A-Z]{2,}\b", text))


def count_swear_words(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0

    from better_profanity import profanity

    profanity.load_censor_words()
    words = text.lower().split()
    return sum(profanity.contains_profanity(word) for word in words)


def count_grammar_errors(text: str, tool) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    matches = tool.check(text)
    return len(matches)


def _sleep_backoff(attempt: int, cap_seconds: int = 60) -> int:
    wait_seconds = min(2 ** max(attempt - 1, 0), cap_seconds)
    time.sleep(wait_seconds)
    return wait_seconds


def _request_with_retries(method, url: str, *, max_retries: int, timeout: int, **kwargs):
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return method(url, timeout=timeout, **kwargs)
        except requests.exceptions.RequestException as e:
            last_exc = e
            _sleep_backoff(attempt, cap_seconds=30)
    raise last_exc


class Command(BaseCommand):
    help = 'Fetch latest tweets using X REST API'

    def add_arguments(self, parser):
        parser.add_argument(
            "--query",
            type=str,
            default="lang:en",
        )
        parser.add_argument(
            "--max-results",
            type=int,
            default=100,
        )
        parser.add_argument(
            "--count",
            type=int,
            default=100,
        )
        parser.add_argument(
            "--wipe",
            action="store_true",
        )
        parser.add_argument(
            "--one-shot",
            action="store_true",
        )
        parser.add_argument(
            "--with-grammar",
            action="store_true",
        )
        parser.add_argument(
            "--max-retries",
            type=int,
            default=8,
        )

    def handle(self, *args, **options):
        API_KEY = settings.TWITTER_API_KEY
        API_SECRET = settings.TWITTER_API_SECRET

        query = str(options.get("query") or "lang:en")
        # X API doesn't allow lang/is/has operators as standalone queries.
        # Normalize minimal case like "lang:en" -> "the lang:en".
        q = query.strip()
        if re.fullmatch(r"lang:[A-Za-z-]+", q):
            query = f"the {q}"
        max_results = int(options.get("max_results") or options.get("max-results") or 100)
        target_count = int(options.get("count") or 200)
        wipe = bool(options.get("wipe"))
        one_shot = bool(options.get("one_shot") or options.get("one-shot"))
        with_grammar = bool(options.get("with_grammar") or options.get("with-grammar"))
        max_retries = int(options.get("max_retries") or options.get("max-retries") or 8)
        if max_results < 10:
            max_results = 10
        if max_results > 100:
            max_results = 100

        if target_count < 1:
            target_count = 1

        tool = None
        if with_grammar:
            import language_tool_python

            tool = language_tool_python.LanguageTool('en-US')

        try:
            # Get bearer token
            auth_url = "https://api.twitter.com/oauth2/token"
            if one_shot:
                auth_response = requests.post(
                    auth_url,
                    auth=(API_KEY, API_SECRET),
                    data={"grant_type": "client_credentials"},
                    timeout=20,
                )
            else:
                auth_response = _request_with_retries(
                    requests.post,
                    auth_url,
                    max_retries=max_retries,
                    timeout=20,
                    auth=(API_KEY, API_SECRET),
                    data={"grant_type": "client_credentials"},
                )
            
            if auth_response.status_code != 200:
                self.stdout.write(self.style.ERROR(f'Failed to get bearer token: {auth_response.status_code}'))
                return
                
            bearer_token = auth_response.json()['access_token']
            
            # Search recent tweets using REST API
            search_url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {
                "Authorization": f"Bearer {bearer_token}",
                "Content-Type": "application/json"
            }
            
            params = {
                "query": query,
                "max_results": max_results,
                "tweet.fields": "created_at,public_metrics,author_id,entities,attachments",
                "expansions": "author_id,attachments.media_keys",
                "user.fields": "name,username,profile_image_url,public_metrics,verified",
                "media.fields": "type",
            }

            if one_shot:
                # Force simplest behavior:
                # - never wipe local DB
                # - single request only (no next_token)
                # - no retry/backoff/wait
                wipe = False
                max_results = 100
                target_count = 100
                params["max_results"] = 100

                response = requests.get(search_url, headers=headers, params=params, timeout=20)
                if response.status_code != 200:
                    self.stdout.write(self.style.ERROR(f'API Error: {response.status_code} - {response.text}'))
                    return

                data = response.json() or {}
                tweets = data.get('data', []) or []
                includes = data.get('includes', {}) or {}
                users = {user['id']: user for user in (includes.get('users') or []) if 'id' in user}
                media_by_key = {m['media_key']: m for m in (includes.get('media') or []) if 'media_key' in m}

                added = 0
                for tweet in tweets:
                    if added >= target_count:
                        break

                    tid = str(tweet.get('id'))
                    if not tid:
                        continue

                    if TweetPost.objects.filter(tweet_id=tid).exists():
                        continue

                    author_id = tweet.get('author_id')
                    user = users.get(author_id, {})

                    created_at_raw = tweet.get('created_at')
                    if not created_at_raw:
                        continue
                    created_at = datetime.fromisoformat(created_at_raw.replace('Z', '+00:00'))
                    posted_hour = created_at.astimezone().hour

                    entities = tweet.get('entities') or {}
                    hashtags = entities.get('hashtags') or []
                    mentions = entities.get('mentions') or []

                    hashtags_count = len(hashtags) if isinstance(hashtags, list) else 0
                    tagged_users_count = len(mentions) if isinstance(mentions, list) else 0

                    photos_count = 0
                    videos_count = 0
                    attachments = tweet.get('attachments') or {}
                    media_keys = attachments.get('media_keys') or []
                    if isinstance(media_keys, list):
                        for k in media_keys:
                            m = media_by_key.get(k) or {}
                            media_type = (m.get('type') or '').lower()
                            if media_type == 'photo':
                                photos_count += 1
                            elif media_type in {'video', 'animated_gif'}:
                                videos_count += 1

                    user_metrics = user.get('public_metrics') or {}
                    followers_count = int(user_metrics.get('followers_count') or 0)
                    following_count = int(user_metrics.get('following_count') or 0)
                    posts_count = int(user_metrics.get('tweet_count') or 0)
                    is_verified = bool(user.get('verified') or False)

                    text = tweet.get('text') or ''
                    description_error_count = count_grammar_errors(text, tool) if tool else 0

                    TweetPost.objects.create(
                        tweet_id=tid,
                        username=user.get('name', 'Unknown'),
                        screen_name=user.get('username', 'unknown'),
                        content=text,
                        created_at=created_at,
                        retweet_count=(tweet.get('public_metrics') or {}).get('retweet_count', 0),
                        like_count=(tweet.get('public_metrics') or {}).get('like_count', 0),
                        profile_image_url=user.get('profile_image_url', ''),
                        posted_hour=posted_hour,
                        photos_count=photos_count,
                        videos_count=videos_count,
                        hashtags_count=hashtags_count,
                        tagged_users_count=tagged_users_count,
                        followers_count=followers_count,
                        posts_count=posts_count,
                        following_count=following_count,
                        is_verified=is_verified,
                        description_error_count=description_error_count,
                        swear_word_count=count_swear_words(text),
                        word_count=count_words(text),
                        emoji_count=count_emojis(text),
                        uppercase_word_count=count_uppercase_words(text),
                    )
                    added += 1

                total_tweets = TweetPost.objects.count()
                self.stdout.write(self.style.SUCCESS(f'Added {added} new tweets. Total in database: {total_tweets}'))
                return
            
            next_token = None
            added = 0
            api_calls = 0

            # Preflight: never wipe local DB unless we can successfully fetch at least one page.
            preflight_done = False
            prefetched_data = None
            if wipe:
                retry = 0
                while True:
                    try:
                        resp = _request_with_retries(
                            requests.get,
                            search_url,
                            max_retries=max_retries,
                            timeout=20,
                            headers=headers,
                            params=params,
                        )
                        api_calls += 1
                    except requests.exceptions.RequestException as e:
                        self.stdout.write(
                            self.style.ERROR(
                                f"Preflight failed due to network error. Aborting without wiping local DB: {e}"
                            )
                        )
                        return

                    if resp.status_code == 200:
                        break

                    if resp.status_code == 429:
                        body = resp.text or ""
                        if "UsageCapExceeded" in body:
                            self.stdout.write(
                                self.style.ERROR(
                                    "X API monthly usage cap exceeded (429 UsageCapExceeded). Aborting without wiping local DB."
                                )
                            )
                            return

                        if retry >= max_retries:
                            self.stdout.write(
                                self.style.ERROR(
                                    f"Preflight failed: 429 Too Many Requests. Exceeded max retries ({max_retries}). Aborting without wiping local DB."
                                )
                            )
                            return

                        retry += 1
                        reset_header = resp.headers.get("x-rate-limit-reset")
                        wait_seconds = min(60 * (2 ** (retry - 1)), 15 * 60)
                        if reset_header:
                            try:
                                reset_ts = int(reset_header)
                                now_ts = int(time.time())
                                wait_seconds = max(1, min(reset_ts - now_ts, 15 * 60))
                            except Exception:
                                pass

                        self.stdout.write(
                            self.style.WARNING(
                                f"Preflight 429 rate limit. Waiting {wait_seconds}s then retrying (attempt {retry}/{max_retries})..."
                            )
                        )
                        time.sleep(wait_seconds)
                        continue

                    self.stdout.write(
                        self.style.ERROR(
                            f"Preflight failed (status={resp.status_code}). Aborting without wiping local DB: {resp.text}"
                        )
                    )
                    return

                prefetched_data = resp.json() or {}
                prefetched_tweets = prefetched_data.get('data', []) or []
                if not prefetched_tweets:
                    self.stdout.write(
                        self.style.ERROR(
                            "Preflight succeeded but returned no tweets. Aborting without wiping local DB."
                        )
                    )
                    return

                # Backup db.sqlite3 before wiping
                db_path = Path(__file__).resolve().parents[3] / 'db.sqlite3'
                if db_path.exists():
                    ts = int(time.time())
                    backup_path = db_path.with_name(f"db.sqlite3.bak.{ts}")
                    shutil.copy2(db_path, backup_path)
                    self.stdout.write(self.style.WARNING(f"DB backup created: {backup_path}"))

                SavedPost.objects.all().delete()
                TweetPost.objects.all().delete()

                preflight_done = True

            while added < target_count:
                if next_token:
                    params['next_token'] = next_token
                else:
                    params.pop('next_token', None)

                retry = 0
                while True:
                    # If we already preflighted for --wipe, reuse that first page after wiping.
                    if preflight_done and prefetched_data is not None and next_token is None:
                        response = None
                        data = prefetched_data
                        prefetched_data = None
                        break

                    try:
                        response = _request_with_retries(
                            requests.get,
                            search_url,
                            max_retries=max_retries,
                            timeout=20,
                            headers=headers,
                            params=params,
                        )
                        api_calls += 1
                    except requests.exceptions.RequestException as e:
                        self.stdout.write(self.style.WARNING(f"Network error while fetching tweets: {e}"))
                        if retry >= max_retries:
                            self.stdout.write(
                                self.style.ERROR(
                                    f"Exceeded max retries ({max_retries}) due to network errors. Stopping."
                                )
                            )
                            return
                        retry += 1
                        _sleep_backoff(retry, cap_seconds=60)
                        continue

                    if response.status_code == 200:
                        break

                    # Handle rate limiting
                    if response.status_code == 429:
                        body = response.text or ""
                        if "UsageCapExceeded" in body:
                            self.stdout.write(
                                self.style.ERROR(
                                    "X API monthly usage cap exceeded (429 UsageCapExceeded). Cannot fetch more tweets."
                                )
                            )
                            return

                        if retry >= max_retries:
                            self.stdout.write(
                                self.style.ERROR(
                                    f"X API 429 Too Many Requests. Exceeded max retries ({max_retries})."
                                )
                            )
                            return

                        retry += 1
                        reset_header = response.headers.get("x-rate-limit-reset")
                        wait_seconds = min(60 * (2 ** (retry - 1)), 15 * 60)
                        if reset_header:
                            try:
                                reset_ts = int(reset_header)
                                now_ts = int(time.time())
                                wait_seconds = max(1, min(reset_ts - now_ts, 15 * 60))
                            except Exception:
                                pass

                        self.stdout.write(
                            self.style.WARNING(
                                f"429 rate limit. Waiting {wait_seconds}s then retrying (attempt {retry}/{max_retries})..."
                            )
                        )
                        time.sleep(wait_seconds)
                        continue

                    # Other errors
                    self.stdout.write(self.style.ERROR(f'API Error: {response.status_code} - {response.text}'))
                    return

                if response is not None:
                    data = response.json() or {}
                tweets = data.get('data', []) or []
                includes = data.get('includes', {}) or {}
                users = {user['id']: user for user in (includes.get('users') or []) if 'id' in user}
                media_by_key = {m['media_key']: m for m in (includes.get('media') or []) if 'media_key' in m}

                if not tweets:
                    break

                for tweet in tweets:
                    if added >= target_count:
                        break

                    tid = str(tweet.get('id'))
                    if not tid:
                        continue

                    if TweetPost.objects.filter(tweet_id=tid).exists():
                        continue

                    author_id = tweet.get('author_id')
                    user = users.get(author_id, {})

                    created_at_raw = tweet.get('created_at')
                    if not created_at_raw:
                        continue
                    created_at = datetime.fromisoformat(created_at_raw.replace('Z', '+00:00'))
                    posted_hour = created_at.astimezone().hour

                    entities = tweet.get('entities') or {}
                    hashtags = entities.get('hashtags') or []
                    mentions = entities.get('mentions') or []

                    hashtags_count = len(hashtags) if isinstance(hashtags, list) else 0
                    tagged_users_count = len(mentions) if isinstance(mentions, list) else 0

                    photos_count = 0
                    videos_count = 0
                    attachments = tweet.get('attachments') or {}
                    media_keys = attachments.get('media_keys') or []
                    if isinstance(media_keys, list):
                        for k in media_keys:
                            m = media_by_key.get(k) or {}
                            media_type = (m.get('type') or '').lower()
                            if media_type == 'photo':
                                photos_count += 1
                            elif media_type in {'video', 'animated_gif'}:
                                videos_count += 1

                    user_metrics = user.get('public_metrics') or {}
                    followers_count = int(user_metrics.get('followers_count') or 0)
                    following_count = int(user_metrics.get('following_count') or 0)
                    posts_count = int(user_metrics.get('tweet_count') or 0)
                    is_verified = bool(user.get('verified') or False)

                    text = tweet.get('text') or ''
                    description_error_count = count_grammar_errors(text, tool) if tool else 0

                    TweetPost.objects.create(
                        tweet_id=tid,
                        username=user.get('name', 'Unknown'),
                        screen_name=user.get('username', 'unknown'),
                        content=text,
                        created_at=created_at,
                        retweet_count=(tweet.get('public_metrics') or {}).get('retweet_count', 0),
                        like_count=(tweet.get('public_metrics') or {}).get('like_count', 0),
                        profile_image_url=user.get('profile_image_url', ''),
                        posted_hour=posted_hour,
                        photos_count=photos_count,
                        videos_count=videos_count,
                        hashtags_count=hashtags_count,
                        tagged_users_count=tagged_users_count,
                        followers_count=followers_count,
                        posts_count=posts_count,
                        following_count=following_count,
                        is_verified=is_verified,
                        description_error_count=description_error_count,
                        swear_word_count=count_swear_words(text),
                        word_count=count_words(text),
                        emoji_count=count_emojis(text),
                        uppercase_word_count=count_uppercase_words(text),
                    )
                    added += 1

                meta = data.get('meta') or {}
                next_token = meta.get('next_token')
                if not next_token:
                    break

            total_tweets = TweetPost.objects.count()
            self.stdout.write(
                self.style.SUCCESS(
                    f'Added {added} new tweets (api_calls={api_calls}). Total in database: {total_tweets}'
                )
            )
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error: {str(e)}'))
