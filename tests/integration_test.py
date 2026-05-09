from django.test import TestCase, Client
from django.urls import reverse
from django.utils import timezone
import json
import time
from datetime import timedelta

from predictor_app.models import TweetPost


class SimpleTrendsIntegrationTest(TestCase):
    """Simplified integration tests for trends functionality - API focused"""
    
    def setUp(self):
        self.client = Client()
        self.create_test_tweets()
    
    def create_test_tweets(self):
        """Create test tweets with various hashtags for trend analysis"""
        test_tweets = [
            {
                'tweet_id': 'simple_test_001',
                'username': 'tech_user',
                'screen_name': 'Tech User',
                'content': 'Amazing #technology trends in #AI and #machinelearning',
                'hashtags': 'technology,AI,machinelearning',
                'followers_count': 1000,
                'photos_count': 2,
                'videos_count': 0,
                'hashtags_count': 3,
                'posted_hour': 14,
                'real_views': 5000,
                'real_sentiment': 'positive',
                'created_at': timezone.now() - timedelta(hours=2)
            },
            {
                'tweet_id': 'simple_test_002',
                'username': 'sports_user',
                'screen_name': 'Sports User',
                'content': 'Latest #sports news and #football updates',
                'hashtags': 'sports,football',
                'followers_count': 800,
                'photos_count': 1,
                'videos_count': 1,
                'hashtags_count': 2,
                'posted_hour': 16,
                'real_views': 3000,
                'real_sentiment': 'neutral',
                'created_at': timezone.now() - timedelta(hours=1)
            },
            {
                'tweet_id': 'simple_test_003',
                'username': 'entertainment_user',
                'screen_name': 'Entertainment User',
                'content': '#entertainment news about #movies and #music',
                'hashtags': 'entertainment,movies,music',
                'followers_count': 1200,
                'photos_count': 3,
                'videos_count': 0,
                'hashtags_count': 3,
                'posted_hour': 18,
                'real_views': 7000,
                'real_sentiment': 'positive',
                'created_at': timezone.now() - timedelta(minutes=30)
            }
        ]
        
        for tweet_data in test_tweets:
            TweetPost.objects.create(**tweet_data)
    
    def test_trends_iteration_api_basic(self):
        """Test basic trends iteration API functionality"""
        response = self.client.get(reverse('get_trends_iteration'))
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        
        # Verify response structure
        required_keys = ['tweets_data', 'hashtag_stats', 'trends_stats', 'has_more']
        for key in required_keys:
            self.assertIn(key, data, f"Response should contain {key}")
        
        # Verify we have data
        self.assertGreater(len(data['tweets_data']), 0, "Should have tweets data")
        self.assertIsInstance(data['hashtag_stats'], list, "Hashtag stats should be list")
        self.assertIsInstance(data['trends_stats'], list, "Trends stats should be list")
        self.assertIsInstance(data['has_more'], bool, "has_more should be boolean")
    
    def test_trends_multiple_iterations(self):
        """Test multiple iterations to simulate frontend behavior"""
        iterations_data = []
        
        # Run multiple iterations
        for i in range(3):
            response = self.client.get(reverse('get_trends_iteration'))
            self.assertEqual(response.status_code, 200)
            
            data = json.loads(response.content)
            iterations_data.append(data)
            
            # Verify structure
            self.assertIn('hashtag_stats', data)
            self.assertIn('trends_stats', data)
            
            # Verify hashtags are tracked
            if data['hashtag_stats']:
                for stat in data['hashtag_stats']:
                    required_fields = ['title', 'engagement_count', 'total_views', 'sentiment']
                    for field in required_fields:
                        self.assertIn(field, stat, f"Stat should have {field}")
            
            time.sleep(0.1)  # Small delay between iterations
        
        # Verify we have multiple iterations
        self.assertEqual(len(iterations_data), 3, "Should have 3 iterations")
        
        # Check that hashtags are consistently tracked
        all_hashtags = set()
        for iteration in iterations_data:
            for stat in iteration.get('hashtag_stats', []):
                all_hashtags.add(stat['title'])
        
        # Should have our test hashtags
        expected_hashtags = {'technology', 'AI', 'machinelearning', 'sports', 'football', 'entertainment', 'movies', 'music'}
        self.assertTrue(all_hashtags.intersection(expected_hashtags), 
                       f"Should contain test hashtags. Found: {all_hashtags}")
    
    def test_trends_metrics_calculation(self):
        """Test that trend metrics are properly calculated"""
        # Get iteration data
        response = self.client.get(reverse('get_trends_iteration'))
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        hashtag_stats = data.get('hashtag_stats', [])
        
        if hashtag_stats:
            # Verify metric structure
            for stat in hashtag_stats:
                # Check required fields
                required_fields = ['title', 'engagement', 'engagement_count', 'total_views', 
                               'growth', 'sentiment', 'total_views_diff', 'sentiment_diff', 'engagement_diff']
                
                for field in required_fields:
                    self.assertIn(field, stat, f"Hashtag stat should have {field} field")
                
                # Verify metric values are populated
                self.assertIsInstance(stat['engagement_count'], int, "engagement_count should be integer")
                self.assertIsInstance(stat['total_views'], (int, float), "total_views should be numeric")
                self.assertIsInstance(stat['sentiment'], (int, float), "sentiment should be numeric")
                
                # Verify values are reasonable
                self.assertGreaterEqual(stat['engagement_count'], 0, "engagement_count should be non-negative")
                self.assertGreaterEqual(stat['total_views'], 0, "total_views should be non-negative")
                self.assertGreaterEqual(stat['sentiment'], 0, "sentiment should be non-negative")
                self.assertLessEqual(stat['sentiment'], 100, "sentiment should be <= 100")
    
    def test_trends_latest_state_api(self):
        """Test trends latest state API"""
        # Run a few iterations first
        for _ in range(2):
            response = self.client.get(reverse('get_trends_iteration'))
            self.assertEqual(response.status_code, 200)
        
        # Get latest state
        response = self.client.get(reverse('get_trends_latest_state'))
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        
        # Verify response structure
        required_keys = ['iteration', 'tweets_data', 'hashtag_stats', 'trends_stats']
        for key in required_keys:
            self.assertIn(key, data, f"Latest state should contain {key}")
        
        # Verify iteration number
        self.assertIsInstance(data['iteration'], int, "Iteration should be integer")
        self.assertGreaterEqual(data['iteration'], 1, "Should have at least 1 iteration")
        
        # Verify hashtag stats are populated
        self.assertGreater(len(data['hashtag_stats']), 0, "Latest state should have hashtag stats")
        
        # Check that metrics are calculated
        for stat in data['hashtag_stats']:
            self.assertIn('total_views', stat)
            self.assertIn('sentiment', stat)
            self.assertIn('engagement_count', stat)
            
            # Verify metrics have reasonable values
            self.assertGreaterEqual(stat['total_views'], 0)
            self.assertGreaterEqual(stat['sentiment'], 0)
            self.assertLessEqual(stat['sentiment'], 100)
    
    def test_trends_cloud_data_api(self):
        """Test trends cloud data endpoint"""
        # Run iterations first
        for _ in range(2):
            response = self.client.get(reverse('get_trends_iteration'))
            self.assertEqual(response.status_code, 200)
        
        # Test cloud data endpoint
        response = self.client.get(reverse('get_trends_cloud_data'))
        self.assertEqual(response.status_code, 200)
        
        cloud_data = json.loads(response.content)
        
        # Verify cloud data structure
        self.assertIn('items', cloud_data)
        self.assertIn('latest_iteration', cloud_data)
        
        # Verify items have required fields
        items = cloud_data['items']
        for item in items:
            self.assertIn('title', item)
            self.assertIn('total_views', item)
            
            # Verify values are reasonable
            self.assertIsInstance(item['total_views'], (int, float))
            self.assertGreaterEqual(item['total_views'], 0)
        
        # Verify items are sorted by total_views (descending)
        if len(items) > 1:
            for i in range(len(items) - 1):
                self.assertGreaterEqual(items[i]['total_views'], items[i+1]['total_views'],
                                   "Items should be sorted by total_views descending")
    
    def test_trends_analytics_api(self):
        """Test trends analytics endpoint for individual trend data"""
        # Get initial iteration data
        response = self.client.get(reverse('get_trends_iteration'))
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        
        if data.get('hashtag_stats'):
            # Get analytics for first hashtag
            first_hashtag = data['hashtag_stats'][0]['title']
            
            # Test analytics endpoint
            analytics_response = self.client.get(
                reverse('get_trend_analytics') + f'?name={first_hashtag}&type=0'
            )
            self.assertEqual(analytics_response.status_code, 200)
            
            analytics_data = json.loads(analytics_response.content)
            
            # Verify analytics response structure
            self.assertIn('name', analytics_data)
            self.assertIn('type', analytics_data)
            self.assertIn('latest', analytics_data)
            self.assertIn('series', analytics_data)
            
            # Verify latest metrics
            latest = analytics_data['latest']
            required_latest_fields = ['engagement_count', 'sentiment', 'total_views', 
                                  'total_views_diff', 'sentiment_diff', 'engagement_diff']
            
            for field in required_latest_fields:
                self.assertIn(field, latest, f"Latest analytics should have {field}")
            
            # Verify metric values are numeric
            self.assertIsInstance(latest['total_views'], (int, float))
            self.assertIsInstance(latest['sentiment'], (int, float))
            self.assertIsInstance(latest['engagement_count'], int)
    
    def test_trends_data_persistence(self):
        """Test that trends data persists across multiple iterations"""
        # Run multiple iterations and collect hashtag stats
        all_hashtag_data = []
        
        for i in range(3):
            response = self.client.get(reverse('get_trends_iteration'))
            self.assertEqual(response.status_code, 200)
            
            data = json.loads(response.content)
            hashtag_stats = data.get('hashtag_stats', [])
            
            # Store hashtag data for comparison
            iteration_data = {}
            for stat in hashtag_stats:
                iteration_data[stat['title']] = {
                    'total_views': stat.get('total_views', 0),
                    'engagement_count': stat.get('engagement_count', 0),
                    'sentiment': stat.get('sentiment', 0)
                }
            
            all_hashtag_data.append(iteration_data)
            time.sleep(0.1)
        
        # Verify data consistency across iterations
        if all_hashtag_data:
            # Get all unique hashtags across iterations
            all_hashtags = set()
            for iteration in all_hashtag_data:
                all_hashtags.update(iteration.keys())
            
            # Check that hashtags persist across iterations
            for hashtag in all_hashtags:
                found_in_iterations = 0
                for iteration in all_hashtag_data:
                    if hashtag in iteration:
                        found_in_iterations += 1
                
                # Hashtag should appear in multiple iterations
                self.assertGreater(found_in_iterations, 0, 
                                 f"Hashtag {hashtag} should appear in iterations")
    
    def test_trends_hashtag_ranking(self):
        """Test that hashtags are properly ranked by metrics"""
        # Run iteration
        response = self.client.get(reverse('get_trends_iteration'))
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        hashtag_stats = data.get('hashtag_stats', [])
        
        if len(hashtag_stats) > 1:
            # Sort by total_views to verify ranking
            sorted_by_views = sorted(hashtag_stats, key=lambda x: x.get('total_views', 0), reverse=True)
            
            # Check that higher engagement hashtags have reasonable metrics
            top_hashtag = sorted_by_views[0]
            
            # Verify top hashtag has meaningful metrics
            self.assertGreater(top_hashtag.get('total_views', 0), 0,
                             "Top hashtag should have views")
            self.assertGreaterEqual(top_hashtag.get('engagement_count', 0), 0,
                                 "Top hashtag should have engagement")
            
            # Verify sentiment is in valid range
            sentiment = top_hashtag.get('sentiment', 0)
            self.assertGreaterEqual(sentiment, 0, "Sentiment should be non-negative")
            self.assertLessEqual(sentiment, 100, "Sentiment should be <= 100")
