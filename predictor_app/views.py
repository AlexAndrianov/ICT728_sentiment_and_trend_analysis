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

logger = logging.getLogger("predictor_app")


def tweet_list(request):
    tweets = TweetPost.objects.all()

    saved_posts = list(SavedPost.objects.select_related('original_tweet').all())
    
    context = {
        'tweets': tweets,
        'title': 'Latest Twitter Posts',
        'saved_posts': saved_posts,
    }
    
    return render(request, 'predictor_app/tweet_list.html', context)


def landing(request):
    """Landing page for social media analytics platform"""
    return render(request, 'predictor_app/landing.html')


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


@require_http_methods(["GET"])
def sentiment(request, tweet_id: int):
    try:
        tweet_id_int = int(tweet_id)

        model_id = _get_selected_sentiment_model_id()
        cache_key = (tweet_id_int, model_id)

        with _SENTIMENT_LOCK:
            fut = _SENTIMENT_FUTURES.get(cache_key)
            if fut is None or fut.cancelled():
                def _job(tid: int, mid: int) -> dict:
                    t0 = time.time()
                    logger.warning("sentiment_view: job_start tweet_id=%s model_id=%s", tid, mid)
                    tweet = TweetPost.objects.get(id=tid)
                    try:
                        mod = importlib.import_module("predictor_app.sentiment_predictor")
                    except Exception as ie:
                        logger.exception("sentiment_view: failed to import predictor_app.sentiment_predictor")
                        raise

                    predict_fn = getattr(mod, "predict_sentiment", None)
                    if predict_fn is None:
                        attrs = [a for a in dir(mod) if not a.startswith("_")]
                        logger.error(
                            "sentiment_view: sentiment_predictor module has no predict_sentiment. attrs=%s",
                            attrs[:80],
                        )
                        raise ImportError(
                            f"predict_sentiment not found in predictor_app.sentiment_predictor; available={attrs[:20]}"
                        )

                    res = predict_fn(tweet.content or "", mid)
                    dt_ms = int((time.time() - t0) * 1000)
                    logger.warning(
                        "sentiment_view: job_done tweet_id=%s model_id=%s score=%s label=%s took_ms=%s",
                        tid,
                        mid,
                        int(res.score),
                        str(res.label),
                        dt_ms,
                    )
                    return {"score": int(res.score), "label": str(res.label)}

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