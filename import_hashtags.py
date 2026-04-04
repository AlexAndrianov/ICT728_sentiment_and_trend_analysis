#!/usr/bin/env python
"""
Script to import hashtags from CSV to TweetPost model
"""

import os
import sys
import django
import re
import csv

# Setup Django
sys.path.append('/Users/alexander/Desktop/KOI/ICT728/ICT728_project')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'predictor.settings')
django.setup()

from predictor_app.models import TweetPost

def extract_hashtags_from_text(text):
    """Extract hashtags from tweet text using regex"""
    if not text:
        return []
    
    # Find all hashtags (words starting with #)
    hashtags = re.findall(r'#\w+', text)
    
    # Remove # and convert to list
    hashtags = [tag[1:] for tag in hashtags]  # Remove # symbol
    
    return hashtags

def generate_hashtags_with_yake(text):
    """Generate hashtags using YAKE library"""
    try:
        import yake
        kw_extractor = yake.KeywordExtractor(
            lan="en",
            n=3,
            dedupLim=0.7,
            top=5,
            features=None
        )
        keywords = kw_extractor.extract_keywords(text)
        
        # Convert keywords to hashtags (capitalize first letter)
        hashtags = []
        for keyword, score in keywords:
            if len(keyword) > 2:  # Only words longer than 2 characters
                hashtag = keyword.replace(' ', '').capitalize()
                hashtags.append(hashtag)
        
        return hashtags[:5]  # Return top 5
    except ImportError:
        print("YAKE not installed, falling back to simple method")
        return generate_hashtags_simple(text)

def generate_hashtags_simple(text):
    """Simple hashtag generation based on common patterns"""
    if not text:
        return []
    
    text_lower = text.lower()
    generated_hashtags = []
    
    # Simple keyword-based hashtag generation
    keyword_hashtags = {
        'music': ['Music', 'MusicNews', 'Billboard'],
        'sport': ['Sports', 'Football', 'Soccer'],
        'politic': ['Politics', 'News', 'Government'],
        'tech': ['Tech', 'Technology', 'AI'],
        'entertainment': ['Entertainment', 'Movies', 'Celebrity'],
        'busines': ['Business', 'Finance', 'Economy'],
        'social media': ['SocialMedia', 'Twitter', 'X'],
        'news': ['News', 'BreakingNews', 'Headlines'],
        'war': ['War', 'Conflict', 'Peace'],
        'government': ['Government', 'Policy', 'Politics'],
        'celebrity': ['Celebrity', 'Stars', 'Famous'],
        'health': ['Health', 'Medicine', 'Wellness'],
        'education': ['Education', 'Learning', 'Students'],
        'environment': ['Environment', 'Climate', 'Nature'],
        'economy': ['Economy', 'Finance', 'Business'],
        'culture': ['Culture', 'Society', 'Lifestyle'],
        'futebol': ['Futebol', 'Football', 'Soccer'],
        'jogo': ['Jogo', 'Game', 'Match'],
        'time': ['Time', 'Team', 'Sports'],
        'championship': ['Championship', 'League', 'Competition'],
        'globo': ['Globo', 'GloboNews', 'Brazil'],
        'brasil': ['Brasil', 'Brazil', 'Brazilian'],
        'presidente': ['Presidente', 'President', 'Politics'],
        'lula': ['Lula', 'Brazil', 'Politics'],
    }
    
    # Check for keywords and add corresponding hashtags
    for keyword, tags in keyword_hashtags.items():
        if keyword in text_lower:
            generated_hashtags.extend(tags)
    
    return list(set(generated_hashtags))  # Remove duplicates

def import_hashtags_from_csv():
    """Import hashtags from CSV file to database"""
    csv_path = '/Users/alexander/Desktop/KOI/ICT728/ICT728_project/ml_models_design/Twitter- datasets.csv'
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
    print(f"Reading CSV file: {csv_path}")
    tweet_hashtags_map = {}
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            tweet_id = row.get('id', '').strip()
            content = row.get('description', '').strip()
            csv_hashtags = row.get('hashtags', '').strip()
            if not tweet_id or not content:
                continue
            
            # Try to get hashtags from CSV first
            hashtags = []
            
            if csv_hashtags and csv_hashtags != 'null':
                try:
                    # Parse hashtags from CSV (they might be in JSON format)
                    import json
                    if csv_hashtags.startswith('['):
                        hashtags_list = json.loads(csv_hashtags.replace('""', '"'))
                        hashtags = [tag.strip('#') for tag in hashtags_list if tag.strip()]
                except:
                    # Fallback: split by comma
                    hashtags = [tag.strip().strip('#') for tag in csv_hashtags.split(',') if tag.strip()]
            
            # If still no hashtags, generate using YAKE library
            if not hashtags:
                hashtags = generate_hashtags_with_yake(content)
            
            # Limit to top 5 hashtags
            if hashtags:
                hashtags = hashtags[:5]
                tweet_hashtags_map[tweet_id] = ', '.join(hashtags)
                print(f"Tweet {tweet_id}: {hashtags}")
    
    # Update database for existing tweets
    updated_count = 0
    for tweet in TweetPost.objects.all():
        if tweet.tweet_id in tweet_hashtags_map:
            tweet.hashtags = tweet_hashtags_map[tweet.tweet_id]
            tweet.save()
            updated_count += 1
            print(f"Updated TweetPost {tweet.id} (tweet_id: {tweet.tweet_id}) with hashtags: {tweet.hashtags}")
    
    # Generate hashtags for tweets without them
    tweets_without_hashtags = TweetPost.objects.filter(hashtags__isnull=True) | TweetPost.objects.filter(hashtags='')
    print(f"\nGenerating hashtags for {tweets_without_hashtags.count()} tweets without hashtags...")
    
    for tweet in tweets_without_hashtags:
        hashtags = generate_hashtags_with_yake(tweet.content)
        if hashtags:
            hashtags = hashtags[:3]  # Limit to 3 for auto-generated
            tweet.hashtags = ', '.join(hashtags)
            tweet.save()
            print(f"Generated hashtags for TweetPost {tweet.id}: {tweet.hashtags}")
        else:
            # Fallback to generic hashtags
            tweet.hashtags = 'SocialMedia, Twitter, Trends'
            tweet.save()
            print(f"Added generic hashtags for TweetPost {tweet.id}: {tweet.hashtags}")
    
    print(f"\nUpdated {updated_count} tweets from CSV")
    print(f"Generated hashtags for {tweets_without_hashtags.count()} tweets")

if __name__ == '__main__':
    import_hashtags_from_csv()
