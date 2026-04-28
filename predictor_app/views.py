from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from .models import TweetPost, SavedPost

import logging
import time
import importlib

from concurrent.futures import Future, ThreadPoolExecutor
import threading


_FORECAST_EXECUTOR = ThreadPoolExecutor(max_workers=4)
_FORECAST_LOCK = threading.Lock()
_FORECAST_FUTURES: dict[int, Future] = {}

_SENTIMENT_EXECUTOR = ThreadPoolExecutor(max_workers=4)
_SENTIMENT_LOCK = threading.Lock()
_SENTIMENT_FUTURES: dict[tuple[int, int], Future] = {}

# Global variable to track last processed tweet ID for batch processing
_LAST_PROCESSED_TWEET_ID = 0
_TRENDS_ITERATION_LOCK = threading.Lock()
_TRENDS_ITERATION_INDEX = 0

# Global dictionary for caching sentiment and forecast results
# Dictionary<TweetId, {sentiment: {score, label}, forecast_views: int}>
gTwitsAnalysisData: dict[int, dict] = {}
gTwitsAnalysisData_LOCK = threading.Lock()

# Dictionary<iteration_number, Dictionary<hashtag_title, hashtag_stats>>
g_hashtag_stats: dict[int, dict[str, dict]] = {}

# Dictionary<iteration_number, Dictionary<cluster_name, cluster_stats>>
g_clusters_stats: dict[int, dict[str, dict]] = {}

logger = logging.getLogger("predictor_app")


def calculate_hashtag_stat(
    tweets,
    previous_iteration,
    current_iteration,
    predict_views_for_tweet,
):
    import concurrent.futures
    from collections import defaultdict

    hashtag_groups = defaultdict(list)
    for tweet in tweets:
        if tweet.hashtags:
            hashtags_list = [tag.strip() for tag in tweet.hashtags.split(',') if tag.strip()]
            for hashtag in hashtags_list:
                hashtag_groups[hashtag].append(tweet)

    hashtag_stats = []
    for hashtag, tweet_list in hashtag_groups.items():
        # Calculate forecast views in parallel
        def get_tweet_views(tweet):
            try:
                # Check cache first
                with gTwitsAnalysisData_LOCK:
                    if tweet.id in gTwitsAnalysisData:
                        cached_data = gTwitsAnalysisData[tweet.id]
                        if 'forecast_views' in cached_data:
                            return cached_data['forecast_views']

                # Calculate if not in cache
                predicted_views = predict_views_for_tweet(tweet)
                views = int(predicted_views or 0)

                # Cache the result
                with gTwitsAnalysisData_LOCK:
                    if tweet.id not in gTwitsAnalysisData:
                        gTwitsAnalysisData[tweet.id] = {}
                    gTwitsAnalysisData[tweet.id]['forecast_views'] = views

                return views
            except:
                return 0

        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tasks
            future_to_tweet = {executor.submit(get_tweet_views, tweet): tweet for tweet in tweet_list}

            # Collect results
            total_views = 0
            for future in concurrent.futures.as_completed(future_to_tweet):
                try:
                    views = future.result(timeout=10)  # 10 second timeout
                    total_views += views
                except Exception as e:
                    print(f"Error calculating views for tweet: {e}")
                    total_views += 0

        hashtag_stats.append({
            'title': hashtag,
            'engagement': f"{len(tweet_list)} posts",
            'engagement_count': len(tweet_list),
            'total_views': total_views,
            'growth': f"+{len(tweet_list) * 5}%",  # Simplified growth calculation
            'sentiment': None,  # Will be filled after sentiment calculation
            'total_views_diff': 0,
            'sentiment_diff': 0,
            'engagement_diff': 0,
        })

    # Calculate sentiment for each hashtag group
    for hashtag_stat in hashtag_stats:
        hashtag = hashtag_stat['title']
        tweet_list = hashtag_groups.get(hashtag, [])

        sentiment_sum = 0
        tweet_count = len(tweet_list)
        for tweet in tweet_list:
            with gTwitsAnalysisData_LOCK:
                if tweet.id in gTwitsAnalysisData:
                    cached_data = gTwitsAnalysisData[tweet.id]
                    if 'sentiment' in cached_data:
                        label = cached_data['sentiment'].get('label')
                        if label == 'positive':
                            sentiment_sum += 1
                        elif label == 'neutral':
                            sentiment_sum += 0.5
                        elif label == 'negative':
                            sentiment_sum += 0

        sentiment_percentage = 0
        if tweet_count > 0:
            sentiment_percentage = round((sentiment_sum / tweet_count) * 100, 1)
        hashtag_stat['sentiment'] = sentiment_percentage

    prev_iteration_stats = g_hashtag_stats.get(previous_iteration, {})

    for hashtag_stat in hashtag_stats:
        prev_hashtag_stat = prev_iteration_stats.get(hashtag_stat['title'])
        if prev_hashtag_stat:
            previous_total_views = int(prev_hashtag_stat.get('total_views', 0) or 0)
            current_total_views = int(hashtag_stat['total_views'] or 0)
            if previous_total_views > 0:
                hashtag_stat['total_views_diff'] = round(
                    ((current_total_views - previous_total_views) / previous_total_views) * 100,
                    1
                )
            else:
                hashtag_stat['total_views_diff'] = 0
            hashtag_stat['sentiment_diff'] = round(float(hashtag_stat['sentiment'] or 0) - float(prev_hashtag_stat.get('sentiment', 0) or 0), 1)
            hashtag_stat['engagement_diff'] = hashtag_stat['engagement_count'] - int(prev_hashtag_stat.get('engagement_count', 0) or 0)
        else:
            hashtag_stat['total_views_diff'] = 0
            hashtag_stat['sentiment_diff'] = 0
            hashtag_stat['engagement_diff'] = 0

    # Sort by total views (descending)
    hashtag_stats.sort(key=lambda x: x['total_views'], reverse=True)

    g_hashtag_stats[current_iteration] = {
        hashtag_stat['title']: hashtag_stat.copy()
        for hashtag_stat in hashtag_stats
    }

    return hashtag_stats


def calculate_trends_stat(
    tweets,
    tweet_to_cluster,
    previous_iteration,
    current_iteration,
    predict_views_for_tweet,
):
    import concurrent.futures
    from collections import defaultdict

    cluster_groups = defaultdict(list)
    for tweet in tweets:
        cluster_label = (tweet_to_cluster.get(tweet.id) or "").strip()
        if cluster_label:
            cluster_groups[cluster_label].append(tweet)

    trends_stats = []
    for cluster_label, tweet_list in cluster_groups.items():
        def get_tweet_views(tweet):
            try:
                with gTwitsAnalysisData_LOCK:
                    if tweet.id in gTwitsAnalysisData:
                        cached_data = gTwitsAnalysisData[tweet.id]
                        if 'forecast_views' in cached_data:
                            return cached_data['forecast_views']

                predicted_views = predict_views_for_tweet(tweet)
                views = int(predicted_views or 0)

                with gTwitsAnalysisData_LOCK:
                    if tweet.id not in gTwitsAnalysisData:
                        gTwitsAnalysisData[tweet.id] = {}
                    gTwitsAnalysisData[tweet.id]['forecast_views'] = views

                return views
            except:
                return 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_tweet = {executor.submit(get_tweet_views, tweet): tweet for tweet in tweet_list}

            total_views = 0
            for future in concurrent.futures.as_completed(future_to_tweet):
                try:
                    views = future.result(timeout=10)
                    total_views += views
                except Exception as e:
                    print(f"Error calculating views for tweet: {e}")
                    total_views += 0

        trends_stats.append({
            'title': cluster_label,
            'cluster_label': cluster_label,
            'engagement': f"{len(tweet_list)} posts",
            'engagement_count': len(tweet_list),
            'total_views': total_views,
            'growth': f"+{len(tweet_list) * 5}%",
            'sentiment': None,
            'total_views_diff': 0,
            'sentiment_diff': 0,
            'engagement_diff': 0,
        })

    for trend_stat in trends_stats:
        cluster_label = trend_stat['title']
        tweet_list = cluster_groups.get(cluster_label, [])

        sentiment_sum = 0
        tweet_count = len(tweet_list)
        for tweet in tweet_list:
            with gTwitsAnalysisData_LOCK:
                if tweet.id in gTwitsAnalysisData:
                    cached_data = gTwitsAnalysisData[tweet.id]
                    if 'sentiment' in cached_data:
                        label = cached_data['sentiment'].get('label')
                        if label == 'positive':
                            sentiment_sum += 1
                        elif label == 'neutral':
                            sentiment_sum += 0.5
                        elif label == 'negative':
                            sentiment_sum += 0

        sentiment_percentage = 0
        if tweet_count > 0:
            sentiment_percentage = round((sentiment_sum / tweet_count) * 100, 1)
        trend_stat['sentiment'] = sentiment_percentage

    prev_iteration_stats = g_clusters_stats.get(previous_iteration, {})

    for trend_stat in trends_stats:
        prev_trend_stat = prev_iteration_stats.get(trend_stat['title'])
        if prev_trend_stat:
            previous_total_views = int(prev_trend_stat.get('total_views', 0) or 0)
            current_total_views = int(trend_stat['total_views'] or 0)
            if previous_total_views > 0:
                trend_stat['total_views_diff'] = round(
                    ((current_total_views - previous_total_views) / previous_total_views) * 100,
                    1
                )
            else:
                trend_stat['total_views_diff'] = 0
            trend_stat['sentiment_diff'] = round(float(trend_stat['sentiment'] or 0) - float(prev_trend_stat.get('sentiment', 0) or 0), 1)
            trend_stat['engagement_diff'] = trend_stat['engagement_count'] - int(prev_trend_stat.get('engagement_count', 0) or 0)
        else:
            trend_stat['total_views_diff'] = 0
            trend_stat['sentiment_diff'] = 0
            trend_stat['engagement_diff'] = 0

    trends_stats.sort(key=lambda x: x['total_views'], reverse=True)

    g_clusters_stats[current_iteration] = {
        trend_stat['title']: trend_stat.copy()
        for trend_stat in trends_stats
    }

    return trends_stats


def tweet_list(request):
    tweets = TweetPost.objects.all()

    saved_posts = list(SavedPost.objects.select_related('original_tweet').all())
    
    context = {
        'tweets': tweets,
        'title': 'Latest Twitter Posts',
        'saved_posts': saved_posts,
    }
    
    return render(request, 'predictor_app/tweet_list.html', context)

def login(request):
    return render(request, 'predictor_app/login.html')

def landing(request):
    """Landing page for social media analytics platform"""
    return render(request, 'predictor_app/landing.html')

def trends(request):
    """Trends dashboard page"""
    # Reset the iteration counter when page is loaded fresh
    global _LAST_PROCESSED_TWEET_ID, _TRENDS_ITERATION_INDEX, g_hashtag_stats, g_clusters_stats
    with _TRENDS_ITERATION_LOCK:
        _LAST_PROCESSED_TWEET_ID = 0
        _TRENDS_ITERATION_INDEX = 0
        g_hashtag_stats = {}
        g_clusters_stats = {}
    
    # Return empty context - data will be loaded via async iteration
    context = {
        'tweets_json': json.dumps([]),
        'hashtag_stats_json': json.dumps([]),
    }
    
    return render(request, 'predictor_app/trends.html', context)

def define_clusters_for_tweets(tweets: list[TweetPost]) -> dict[int, str]:
    """Define clusters for a list of tweets and return mapping of tweet_id to cluster label."""
    from collections import Counter
    from .trend_predictor import cluster_by_trends, define_cluster_name, define_similar_cluster

    if not tweets:
        return {}

    tweet_texts = [(tweet.content or "").strip() for tweet in tweets]
    tweet_to_cluster_idx = cluster_by_trends(tweet_texts)
    cluster_to_name = define_cluster_name(tweet_to_cluster_idx, tweet_texts)

    current_iteration = _TRENDS_ITERATION_INDEX
    previous_iteration = current_iteration - 1
    previous_cluster_names = list(g_clusters_stats.get(previous_iteration, {}).keys())

    # Reuse old cluster names when semantic similarity is high.
    verified_cluster_names = define_similar_cluster(cluster_to_name, previous_cluster_names)

    tweet_to_cluster_label: dict[int, str] = {}
    labels = []
    for tweet_idx, tweet in enumerate(tweets):
        cluster_idx = int(tweet_to_cluster_idx.get(tweet_idx, tweet_idx))
        label = (verified_cluster_names.get(cluster_idx) or "").strip()
        if not label:
            label = f"cluster-{cluster_idx + 1}"
        tweet_to_cluster_label[tweet.id] = label
        labels.append(label)

    counts = Counter(labels)
    g_clusters_stats[current_iteration] = {
        label: {
            "cluster_label": label,
            "tweet_count": count,
        }
        for label, count in counts.items()
    }

    return tweet_to_cluster_label

@require_http_methods(["GET"])
def get_trends_iteration(request):
    """Get next batch of tweets for trends calculation (50 posts per batch)"""
    global _LAST_PROCESSED_TWEET_ID, _TRENDS_ITERATION_INDEX, g_hashtag_stats
    
    try:
        with _TRENDS_ITERATION_LOCK:
            # Get next batch of 50 tweets (or less if fewer remain)
            tweets = list(TweetPost.objects.filter(id__gt=_LAST_PROCESSED_TWEET_ID).order_by('id')[:50])
            
            if not tweets:
                # No more tweets to process
                return JsonResponse({'tweets_data': [], 'has_more': False})
            
            # Update last processed ID
            last_tweet = tweets[-1]
            _LAST_PROCESSED_TWEET_ID = last_tweet.id
            _TRENDS_ITERATION_INDEX += 1
            current_iteration = _TRENDS_ITERATION_INDEX
            previous_iteration = current_iteration - 1
            tweet_to_cluster = define_clusters_for_tweets(tweets)
            
            # Convert to JSON-safe format
            tweets_data = []
            for tweet in tweets:
                tweets_data.append({
                    'id': tweet.id,
                    'username': tweet.username,
                    'screen_name': tweet.screen_name,
                    'content': tweet.content,
                    'profile_image_url': tweet.profile_image_url,
                    'is_dataset_tweet': tweet.is_dataset_tweet,
                    'real_sentiment': tweet.real_sentiment,
                    'real_views': tweet.real_views,
                    'hashtags': tweet.hashtags,
                    'sentiment': None,  # Will be filled after calculation
                    'forecast_views': None,  # Will be filled after calculation
                    'cluster_label': tweet_to_cluster.get(tweet.id),  # filled from tweet_to_cluster
                })
            
            # Calculate sentiment for all tweets in batch (parallel processing)
            import concurrent.futures
            from .trend_predictor import predict_views_for_tweet
            
            model_id = _get_selected_sentiment_model_id()
            def get_tweet_sentiment(tweet):
                try:
                    return _calculate_sentiment_for_tweet(tweet.id, tweet.content or "", model_id)
                except:
                    return None
            
            # Calculate sentiment for all tweets in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                sentiment_futures = {executor.submit(get_tweet_sentiment, tweet): tweet for tweet in tweets}
                sentiment_results = {}
                for future in concurrent.futures.as_completed(sentiment_futures):
                    try:
                        result = future.result(timeout=10)
                        tweet = sentiment_futures[future]
                        sentiment_results[tweet.id] = result
                    except Exception as e:
                        logger.warning("Error calculating sentiment for tweet: %s", e)
            
            # Fill sentiment data in tweets_data
            for tweet_data in tweets_data:
                if tweet_data['id'] in sentiment_results:
                    sentiment_result = sentiment_results[tweet_data['id']]
                    if sentiment_result:
                        tweet_data['sentiment'] = sentiment_result.get('label')
                        tweet_data['sentiment_score'] = sentiment_result.get('score')
            
            hashtag_stats = calculate_hashtag_stat(
                tweets=tweets,
                previous_iteration=previous_iteration,
                current_iteration=current_iteration,
                predict_views_for_tweet=predict_views_for_tweet,
            )

            trends_stats = calculate_trends_stat(
                tweets=tweets,
                tweet_to_cluster=tweet_to_cluster,
                previous_iteration=previous_iteration,
                current_iteration=current_iteration,
                predict_views_for_tweet=predict_views_for_tweet,
            )
            
            # Fill forecast_views data in tweets_data from cache (after calculations)
            for tweet_data in tweets_data:
                with gTwitsAnalysisData_LOCK:
                    if tweet_data['id'] in gTwitsAnalysisData:
                        cached_data = gTwitsAnalysisData[tweet_data['id']]
                        if 'forecast_views' in cached_data:
                            tweet_data['forecast_views'] = cached_data['forecast_views']
            
            return JsonResponse({
                'tweets_data': tweets_data,
                'hashtag_stats': hashtag_stats,
                'trends_stats': trends_stats,
                'has_more': True,
            })
            
    except Exception as e:
        logger.exception("get_trends_iteration error")
        return JsonResponse({'error': str(e), 'tweets_data': [], 'has_more': False}, status=500)


def _get_trend_stats_map(trend_type: int) -> dict[int, dict[str, dict]]:
    if int(trend_type or 0) == 0:
        return g_hashtag_stats
    return g_clusters_stats


def _find_matching_trend_key(stats_for_iteration: dict[str, dict], trend_name: str) -> str | None:
    if not stats_for_iteration:
        return None

    raw = (trend_name or "").strip()
    if not raw:
        return None

    if raw in stats_for_iteration:
        return raw

    lowered = raw.lower()
    for key in stats_for_iteration.keys():
        if (key or "").strip().lower() == lowered:
            return key

    return None


@require_http_methods(["GET"])
def get_trend_analytics(request):
    trend_name = (request.GET.get("name") or "").strip()
    trend_type_raw = (request.GET.get("type") or "0").strip()

    try:
        trend_type = int(trend_type_raw)
    except ValueError:
        trend_type = 0

    if not trend_name:
        return JsonResponse({"error": "Missing 'name'"}, status=400)

    stats_map = _get_trend_stats_map(trend_type)
    iterations = sorted(stats_map.keys())
    if not iterations:
        return JsonResponse({
            "name": trend_name,
            "type": trend_type,
            "series": [],
            "latest": None,
        })

    latest_iteration = iterations[-1]
    latest_iteration_stats = stats_map.get(latest_iteration, {})
    matching_key = _find_matching_trend_key(latest_iteration_stats, trend_name)
    if matching_key is None:
        for it in reversed(iterations):
            matching_key = _find_matching_trend_key(stats_map.get(it, {}), trend_name)
            if matching_key is not None:
                break

    if matching_key is None:
        return JsonResponse({
            "name": trend_name,
            "type": trend_type,
            "series": [],
            "latest": None,
        })

    series = []
    for it in iterations:
        it_stats = stats_map.get(it, {})
        key_for_iteration = _find_matching_trend_key(it_stats, matching_key) or _find_matching_trend_key(it_stats, trend_name)
        if key_for_iteration is None:
            continue
        stat = it_stats.get(key_for_iteration) or {}
        series.append({
            "iteration": it,
            "forecast_views": int(stat.get("total_views", 0) or 0),
        })

    latest_stat = (latest_iteration_stats.get(matching_key) or {})

    return JsonResponse({
        "name": matching_key,
        "type": trend_type,
        "latest_iteration": latest_iteration,
        "latest": {
            "engagement_count": int(latest_stat.get("engagement_count", 0) or 0),
            "sentiment": float(latest_stat.get("sentiment", 0) or 0),
            "total_views": int(latest_stat.get("total_views", 0) or 0),
            "total_views_diff": float(latest_stat.get("total_views_diff", 0) or 0),
            "sentiment_diff": float(latest_stat.get("sentiment_diff", 0) or 0),
            "engagement_diff": int(latest_stat.get("engagement_diff", 0) or 0),
        },
        "series": series,
    })


def index(request):
    tweets = TweetPost.objects.all()

    saved_posts = list(SavedPost.objects.select_related('original_tweet').all())
    
    context = {
        'tweets': tweets,
        'title': 'Latest Twitter Posts',
        'saved_posts': saved_posts,
    }
    
    return render(request, 'predictor_app/tweet_list.html', context)


@csrf_exempt
@require_http_methods(["POST"])
def save_to_dashboard(request):
    try:
        data = json.loads(request.body)
        tweet_id = data.get('tweet_id')
        
        tweet = TweetPost.objects.get(id=tweet_id)

        existing_for_tweet = SavedPost.objects.filter(original_tweet=tweet).first()
        if existing_for_tweet:
            return JsonResponse({
                'success': True,
                'position': {'x': existing_for_tweet.position_x, 'y': existing_for_tweet.position_y},
                'already_saved': True,
            })
        
        # Find next available position in grid
        existing_posts = SavedPost.objects.all()
        if existing_posts.exists():
            max_y = existing_posts.order_by('-position_y').first().position_y
            posts_in_last_row = existing_posts.filter(position_y=max_y)
            max_x = posts_in_last_row.order_by('-position_x').first().position_x
            
            if max_x < 2:  # 3 columns (0, 1, 2)
                position_x = max_x + 1
                position_y = max_y
            else:
                position_x = 0
                position_y = max_y + 1
        else:
            position_x = 0
            position_y = 0
        
        saved_post = SavedPost.objects.create(
            original_tweet=tweet,
            position_x=position_x,
            position_y=position_y
        )
        
        return JsonResponse({'success': True, 'position': {'x': position_x, 'y': position_y}})
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@require_http_methods(["GET"])
def forecast_views(request, tweet_id: int):
    try:
        tweet_id_int = int(tweet_id)

        # Check cache first
        with gTwitsAnalysisData_LOCK:
            if tweet_id_int in gTwitsAnalysisData:
                cached_data = gTwitsAnalysisData[tweet_id_int]
                if 'forecast_views' in cached_data:
                    logger.info("forecast_view: cache hit tweet_id=%s", tweet_id_int)
                    return JsonResponse({'status': 'ready', 'views': cached_data['forecast_views']})

        with _FORECAST_LOCK:
            fut = _FORECAST_FUTURES.get(tweet_id_int)
            if fut is None or fut.cancelled():
                def _job(tid: int) -> int:
                    tweet = TweetPost.objects.get(id=tid)
                    from .trend_predictor import predict_views_for_tweet

                    return predict_views_for_tweet(tweet)

                fut = _FORECAST_EXECUTOR.submit(_job, tweet_id_int)
                _FORECAST_FUTURES[tweet_id_int] = fut

        if not fut.done():
            logger.warning("forecast_view: pending tweet_id=%s", tweet_id_int)
            return JsonResponse({'status': 'pending'})

        err = fut.exception()
        if err is not None:
            with _FORECAST_LOCK:
                _FORECAST_FUTURES.pop(tweet_id_int, None)
            return JsonResponse({'status': 'error', 'error': str(err)})

        views = int(fut.result() or 0)
        
        # Cache the result
        with gTwitsAnalysisData_LOCK:
            if tweet_id_int not in gTwitsAnalysisData:
                gTwitsAnalysisData[tweet_id_int] = {}
            gTwitsAnalysisData[tweet_id_int]['forecast_views'] = views
        
        return JsonResponse({'status': 'ready', 'views': views})
    except Exception as e:
        return JsonResponse({'status': 'error', 'error': str(e)})


def _get_selected_sentiment_model_id() -> int:
    try:
        from .models import SentimentModelSettings

        obj = SentimentModelSettings.objects.order_by("id").first()
        if obj and obj.selected_model_id:
            return int(obj.selected_model_id)
    except Exception:
        pass
    return 1


def _calculate_sentiment_for_tweet(tweet_id: int, tweet_content: str, model_id: int) -> dict:
    """Shared function to calculate sentiment for a tweet and cache the result."""
    # Check cache first
    with gTwitsAnalysisData_LOCK:
        if tweet_id in gTwitsAnalysisData:
            cached_data = gTwitsAnalysisData[tweet_id]
            if 'sentiment' in cached_data:
                return cached_data['sentiment']
    
    # Calculate if not in cache
    try:
        mod = importlib.import_module("predictor_app.nltk_vader_sentiment")
    except Exception as ie:
        logger.exception("_calculate_sentiment_for_tweet: failed to import predictor_app.nltk_vader_sentiment")
        return None
    
    predict_fn = getattr(mod, "predict_sentiment", None)
    if predict_fn is None:
        attrs = [a for a in dir(mod) if not a.startswith("_")]
        logger.error(
            "_calculate_sentiment_for_tweet: nltk_vader_sentiment module has no predict_sentiment. attrs=%s",
            attrs[:80],
        )
        raise ImportError(
            f"predict_sentiment not found in predictor_app.nltk_vader_sentiment; available={attrs[:20]}"
        )
    
    res = predict_fn(tweet_content or "", model_id)
    sentiment_result = {"score": int(res.score), "label": str(res.label)}
    
    # Cache the result
    with gTwitsAnalysisData_LOCK:
        if tweet_id not in gTwitsAnalysisData:
            gTwitsAnalysisData[tweet_id] = {}
        gTwitsAnalysisData[tweet_id]['sentiment'] = sentiment_result
    
    return sentiment_result


@require_http_methods(["GET"])
def sentiment(request, tweet_id: int):
    try:
        tweet_id_int = int(tweet_id)

        # Check cache first
        with gTwitsAnalysisData_LOCK:
            if tweet_id_int in gTwitsAnalysisData:
                cached_data = gTwitsAnalysisData[tweet_id_int]
                if 'sentiment' in cached_data:
                    logger.info("sentiment_view: cache hit tweet_id=%s", tweet_id_int)
                    return JsonResponse({'status': 'ready', **cached_data['sentiment']})

        model_id = _get_selected_sentiment_model_id()
        cache_key = (tweet_id_int, model_id)

        with _SENTIMENT_LOCK:
            fut = _SENTIMENT_FUTURES.get(cache_key)
            if fut is None or fut.cancelled():
                def _job(tid: int, mid: int) -> dict:
                    t0 = time.time()
                    logger.warning("sentiment_view: job_start tweet_id=%s model_id=%s", tid, mid)
                    tweet = TweetPost.objects.get(id=tid)
                    
                    # Use shared sentiment calculation function
                    result = _calculate_sentiment_for_tweet(tweet.id, tweet.content or "", mid)
                    
                    dt_ms = int((time.time() - t0) * 1000)
                    logger.warning(
                        "sentiment_view: job_done tweet_id=%s model_id=%s score=%s label=%s took_ms=%s",
                        tid,
                        mid,
                        result.get("score") if result else None,
                        result.get("label") if result else None,
                        dt_ms,
                    )
                    return result

                fut = _SENTIMENT_EXECUTOR.submit(_job, tweet_id_int, model_id)
                _SENTIMENT_FUTURES[cache_key] = fut
                logger.warning("sentiment_view: job_submitted tweet_id=%s model_id=%s", tweet_id_int, model_id)

        if not fut.done():
            return JsonResponse({'status': 'pending'})

        err = fut.exception()
        if err is not None:
            with _SENTIMENT_LOCK:
                _SENTIMENT_FUTURES.pop(cache_key, None)
            logger.warning("sentiment_view: error tweet_id=%s model_id=%s err=%r", tweet_id_int, model_id, err)
            return JsonResponse({'status': 'error', 'error': str(err)})

        out = fut.result() or {}
        # Drop completed future to avoid serving stale results after code/model changes.
        with _SENTIMENT_LOCK:
            _SENTIMENT_FUTURES.pop(cache_key, None)
        
        logger.warning(
            "sentiment_view: ready tweet_id=%s model_id=%s score=%s label=%s",
            tweet_id_int,
            model_id,
            out.get("score"),
            out.get("label"),
        )
        return JsonResponse({'status': 'ready', **out})
    except Exception as e:
        logger.exception("sentiment_view: exception tweet_id=%r", tweet_id)
        return JsonResponse({'status': 'error', 'error': str(e)})


@csrf_exempt
@require_http_methods(["POST"])
def remove_from_dashboard(request):
    try:
        data = json.loads(request.body)
        tweet_id = data.get('tweet_id')

        deleted_count, _ = SavedPost.objects.filter(original_tweet_id=tweet_id).delete()
        if deleted_count == 0:
            return JsonResponse({'success': True, 'deleted': 0})
        
        return JsonResponse({'success': True, 'deleted': deleted_count})
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})