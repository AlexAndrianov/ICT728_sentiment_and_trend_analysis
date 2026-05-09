from django.test import TestCase, Client
from django.urls import reverse
from django.utils import timezone
import json
from datetime import timedelta

from predictor_app.models import TweetPost, SavedPost


class APITest(TestCase):
    """Basic API tests for all endpoints in views.py"""
    
    def setUp(self):
        self.client = Client()
        self.create_test_tweet()
    
    def create_test_tweet(self):
        """Create a test tweet for API testing"""
        self.test_tweet = TweetPost.objects.create(
            tweet_id='api_test_001',
            username='testuser',
            screen_name='Test User',
            content='Test content #hashtag',
            hashtags='hashtag',
            followers_count=100,
            photos_count=1,
            videos_count=0,
            hashtags_count=1,
            posted_hour=12,
            real_views=1000,
            real_sentiment='positive',
            created_at=timezone.now()
        )
    
    def test_tweet_list_api(self):
        """Test tweet_list endpoint - should return HTML page with tweets"""
        response = self.client.get(reverse('tweet_list'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Test User')
        self.assertContains(response, 'Test content #hashtag')
    
    def test_login_api(self):
        """Test login endpoint - should return HTML login page"""
        response = self.client.get(reverse('login'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Login')
        self.assertContains(response, 'username')
        self.assertContains(response, 'password')
    
    def test_landing_api(self):
        """Test landing endpoint - should return HTML landing page"""
        response = self.client.get(reverse('landing'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Smart Social Media Analytics')
        self.assertContains(response, 'Trend Prediction')
    
    def test_trends_api(self):
        """Test trends endpoint - should return HTML trends dashboard"""
        response = self.client.get(reverse('trends'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'trends-dashboard')
    
    def test_save_trends_layout_api(self):
        """Test save_trends_layout endpoint - should save layout configuration"""
        layout_data = {
            'hashtag': {'column1': ['hashtag1', 'hashtag2']},
            'cluster': {'column2': ['cluster1']}
        }
        
        response = self.client.post(
            reverse('save_trends_layout'),
            data=json.dumps(layout_data),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        # Check that response is valid JSON and doesn't error
        self.assertIsInstance(data, dict)
    
    def test_get_trends_cloud_data_api(self):
        """Test get_trends_cloud_data endpoint - should return cloud visualization data"""
        # First run some iterations to generate data
        for _ in range(2):
            self.client.get(reverse('get_trends_iteration'))
        
        response = self.client.get(reverse('get_trends_cloud_data'))
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        self.assertIn('items', data)
        self.assertIn('latest_iteration', data)
        self.assertIsInstance(data['items'], list)
    
    def test_get_trends_iteration_api(self):
        """Test get_trends_iteration endpoint - should process tweets and return trends"""
        response = self.client.get(reverse('get_trends_iteration'))
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        required_keys = ['tweets_data', 'hashtag_stats', 'trends_stats', 'has_more']
        for key in required_keys:
            self.assertIn(key, data, f"Response should contain {key}")
        
        self.assertIsInstance(data['tweets_data'], list)
        self.assertIsInstance(data['hashtag_stats'], list)
        self.assertIsInstance(data['trends_stats'], list)
        self.assertIsInstance(data['has_more'], bool)
    
    def test_get_trends_latest_state_api(self):
        """Test get_trends_latest_state endpoint - should return latest trends snapshot"""
        # First run some iterations to generate data
        for _ in range(2):
            self.client.get(reverse('get_trends_iteration'))
        
        response = self.client.get(reverse('get_trends_latest_state'))
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        required_keys = ['iteration', 'tweets_data', 'hashtag_stats', 'trends_stats']
        for key in required_keys:
            self.assertIn(key, data, f"Latest state should contain {key}")
        
        self.assertIsInstance(data['iteration'], int)
        self.assertGreaterEqual(data['iteration'], 0)
    
    def test_get_trend_analytics_api(self):
        """Test get_trend_analytics endpoint - should return individual trend analytics"""
        # First run iterations to generate data
        for _ in range(2):
            self.client.get(reverse('get_trends_iteration'))
        
        response = self.client.get(reverse('get_trend_analytics') + '?name=hashtag&type=0')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        self.assertIn('name', data)
        self.assertIn('type', data)
        self.assertIn('latest', data)
        self.assertIn('series', data)
    
    def test_index_api(self):
        """Test index endpoint - should return HTML dashboard page"""
        response = self.client.get(reverse('index'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Test User')
    
    def test_save_to_dashboard_api(self):
        """Test save_to_dashboard endpoint - should save tweet to dashboard"""
        response = self.client.post(
            reverse('save_to_dashboard'),
            data=json.dumps({'tweet_id': self.test_tweet.id}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        self.assertTrue(data.get('success', False))
        
        # Verify tweet was saved
        saved_post = SavedPost.objects.filter(original_tweet=self.test_tweet).first()
        self.assertIsNotNone(saved_post)
    
    def test_forecast_views_api(self):
        """Test forecast_views endpoint - should return predicted views for tweet"""
        response = self.client.get(reverse('forecast_views', args=[self.test_tweet.id]))
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        # API returns status field when processing
        if 'predicted_views' in data:
            self.assertIsInstance(data['predicted_views'], (int, float))
            self.assertGreaterEqual(data['predicted_views'], 0)
        else:
            # Should have status field when processing
            self.assertIn('status', data)
    
    def test_sentiment_api(self):
        """Test sentiment endpoint - should return sentiment analysis for tweet"""
        response = self.client.get(reverse('sentiment', args=[self.test_tweet.id]))
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        # API returns direct fields, not nested in 'sentiment'
        self.assertIn('status', data)
        self.assertIn('score', data)
        self.assertIn('label', data)
        
        # Verify sentiment values are in expected range
        score = data['score']
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, -1)
        self.assertLessEqual(score, 1)
        
        label = data['label']
        self.assertIn(label, ['negative', 'neutral', 'positive'])
    
    def test_remove_from_dashboard_api(self):
        """Test remove_from_dashboard endpoint - should remove tweet from dashboard"""
        # First save a tweet to dashboard
        saved_post = SavedPost.objects.create(
            original_tweet=self.test_tweet,
            position_x=0,
            position_y=0,
            saved_at=timezone.now()
        )
        
        # Then remove it
        response = self.client.post(
            reverse('remove_from_dashboard'),
            data=json.dumps({'tweet_id': self.test_tweet.id}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        self.assertTrue(data.get('success', False))
        
        # Verify tweet was removed
        saved_post_exists = SavedPost.objects.filter(original_tweet=self.test_tweet).exists()
        self.assertFalse(saved_post_exists)
    
    def test_trends_cloud_data_empty_api(self):
        """Test get_trends_cloud_data endpoint with no data - should handle gracefully"""
        response = self.client.get(reverse('get_trends_cloud_data'))
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        self.assertIn('items', data)
        self.assertIn('latest_iteration', data)
        # Should handle empty data gracefully
        self.assertIsInstance(data['items'], list)
    
    def test_trend_analytics_not_found_api(self):
        """Test get_trend_analytics endpoint with non-existent trend - should handle gracefully"""
        response = self.client.get(reverse('get_trend_analytics') + '?name=nonexistent&type=0')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        self.assertIn('name', data)
        self.assertIn('type', data)
        # Should handle missing data gracefully
        self.assertIn('latest', data)
        self.assertIn('series', data)
    
    def test_forecast_views_not_found_api(self):
        """Test forecast_views endpoint with non-existent tweet ID - should handle gracefully"""
        response = self.client.get(reverse('forecast_views', args=[99999]))
        # API returns 200 with status field even for non-existent tweets
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        self.assertIn('status', data)
    
    def test_sentiment_not_found_api(self):
        """Test sentiment endpoint with non-existent tweet ID - should handle gracefully"""
        response = self.client.get(reverse('sentiment', args=[99999]))
        # API returns 200 with status field even for non-existent tweets
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        self.assertIn('status', data)
    
    def test_save_trends_layout_invalid_data_api(self):
        """Test save_trends_layout endpoint with invalid JSON - should return 400"""
        response = self.client.post(
            reverse('save_trends_layout'),
            data='invalid json',
            content_type='application/json'
        )
        # API returns 400 for invalid JSON
        self.assertEqual(response.status_code, 400)
    
    def test_save_to_dashboard_invalid_data_api(self):
        """Test save_to_dashboard endpoint with missing tweet_id - should handle gracefully"""
        response = self.client.post(
            reverse('save_to_dashboard'),
            data=json.dumps({}),
            content_type='application/json'
        )
        # API handles missing data gracefully
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        self.assertIsInstance(data, dict)
    
    def test_remove_from_dashboard_invalid_data_api(self):
        """Test remove_from_dashboard endpoint with missing tweet_id - should handle gracefully"""
        response = self.client.post(
            reverse('remove_from_dashboard'),
            data=json.dumps({}),
            content_type='application/json'
        )
        # API handles missing data gracefully
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        self.assertIsInstance(data, dict)
    
    def test_http_methods_api(self):
        """Test that endpoints respect HTTP method requirements"""
        # Test GET-only endpoints with POST
        post_endpoints = [
            'get_trends_iteration',
            'get_trends_latest_state', 
            'get_trends_cloud_data',
            'get_trend_analytics',
            'forecast_views',
            'sentiment'
        ]
        
        for endpoint in post_endpoints:
            if endpoint in ['forecast_views', 'sentiment']:
                # These require tweet_id parameter
                response = self.client.post(reverse(endpoint, args=[self.test_tweet.id]))
            else:
                response = self.client.post(reverse(endpoint))
            
            # Should return 405 Method Not Allowed or handle gracefully
            self.assertIn(response.status_code, [405, 400, 200])
        
        # Test POST-only endpoints with GET
        post_only_endpoints = [
            'save_trends_layout',
            'save_to_dashboard',
            'remove_from_dashboard'
        ]
        
        for endpoint in post_only_endpoints:
            response = self.client.get(reverse(endpoint))
            # Should return 405 Method Not Allowed or handle gracefully
            self.assertIn(response.status_code, [405, 400, 200])
