#!/usr/bin/env python3
"""
RUGGUARD Bot - Solana Project Trustworthiness Analyzer

This bot monitors Twitter for mentions and analyzes the trustworthiness of accounts
that post about Solana projects. When someone replies to a tweet with
"@projectrugguard riddle me this", the bot will analyze the original tweet's author
and post a trustworthiness report.
"""

import os
import sys
import time
import json
import sqlite3
import logging
import datetime
from typing import Dict, Optional, List, Any

import tweepy
import requests
from textblob import TextBlob
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rugguard.log')
    ]
)
logger = logging.getLogger('rugguard')

def load_environment() -> bool:
    """Load environment variables and validate configuration."""
    load_dotenv()
    
    required_vars = [
        "API_KEY",
        "API_SECRET",
        "ACCESS_TOKEN",
        "ACCESS_TOKEN_SECRET",
        "BEARER_TOKEN"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    return True

# Load and validate environment
if not load_environment():
    logger.error("Failed to load required environment variables. Exiting.")
    sys.exit(1)

# Twitter API credentials
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

def initialize_twitter_client() -> Optional[tweepy.Client]:
    """Initialize and return the Twitter API client."""
    try:
        # Initialize OAuth1 user context client
        auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        
        # Initialize API v1.1 client (needed for some endpoints)
        api = tweepy.API(auth, wait_on_rate_limit=True)
        
        # Initialize API v2 client
        client = tweepy.Client(
            bearer_token=BEARER_TOKEN,
            consumer_key=API_KEY,
            consumer_secret=API_SECRET,
            access_token=ACCESS_TOKEN,
            access_token_secret=ACCESS_TOKEN_SECRET,
            wait_on_rate_limit=True
        )
        
        # Verify credentials
        api.verify_credentials()
        logger.info("Successfully authenticated with Twitter API")
        return client, api
        
    except Exception as e:
        logger.error(f"Failed to initialize Twitter client: {e}")
        return None, None

# Initialize Twitter clients
client, api = initialize_twitter_client()
if not client or not api:
    logger.error("Failed to initialize Twitter clients. Exiting.")
    sys.exit(1)

def setup_database() -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    """Set up the SQLite database and return connection and cursor."""
    try:
        conn = sqlite3.connect("processed_tweets.db")
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_tweets (
            tweet_id TEXT PRIMARY KEY,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'pending'
        )
        ''')
        
        # Create table for analysis history
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            username TEXT NOT NULL,
            trust_score REAL,
            analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        logger.info("Database setup completed successfully")
        return conn, cursor
        
    except sqlite3.Error as e:
        logger.error(f"Database error during setup: {e}")
        raise

# Initialize database
try:
    conn, cursor = setup_database()
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    sys.exit(1)

# Constants
TRUSTED_ACCOUNTS_URL = "https://raw.githubusercontent.com/devsyrem/turst-list/main/list"
TRUSTED_ACCOUNTS_CACHE = {
    'accounts': [],
    'last_updated': 0
}
CACHE_TTL = 3600  # 1 hour in seconds

def get_trusted_accounts() -> List[str]:
    """
    Fetch the list of trusted accounts from GitHub with caching.
    
    Returns:
        List of trusted account usernames
    """
    current_time = time.time()
    
    # Return cached accounts if still valid
    if (current_time - TRUSTED_ACCOUNTS_CACHE['last_updated']) < CACHE_TTL and TRUSTED_ACCOUNTS_CACHE['accounts']:
        return TRUSTED_ACCOUNTS_CACHE['accounts']
    
    try:
        logger.info("Fetching updated list of trusted accounts")
        response = requests.get(TRUSTED_ACCOUNTS_URL, timeout=10)
        response.raise_for_status()
        
        # Process and cache the accounts
        accounts = [acc.strip() for acc in response.text.strip().split("\n") if acc.strip()]
        TRUSTED_ACCOUNTS_CACHE.update({
            'accounts': accounts,
            'last_updated': current_time
        })
        
        logger.info(f"Successfully updated {len(accounts)} trusted accounts")
        return accounts
        
    except requests.RequestException as e:
        logger.error(f"Error fetching trusted accounts: {e}")
        # Return cached accounts even if stale as fallback
        return TRUSTED_ACCOUNTS_CACHE['accounts'] if TRUSTED_ACCOUNTS_CACHE['accounts'] else []
    except Exception as e:
        logger.error(f"Unexpected error in get_trusted_accounts: {e}")
        return []

def analyze_user(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Analyze a Twitter user's trustworthiness based on various metrics.
    
    Args:
        user_id: Twitter user ID to analyze
        
    Returns:
        Dict containing analysis results or None if analysis fails
    """
    logger.info(f"Starting analysis for user ID: {user_id}")
    
    try:
        # Get basic user information
        user = client.get_user(
            id=user_id,
            user_fields=[
                "created_at",
                "description",
                "public_metrics",
                "verified",
                "profile_image_url",
                "url",
                "location",
                "pinned_tweet_id"
            ]
        )
        
        if not user or not user.data:
            logger.warning(f"No user data found for ID: {user_id}")
            return None
            
        user_data = user.data
        logger.info(f"Retrieved data for user: @{user_data.username}")

        # Calculate account metrics
        current_time = datetime.datetime.now(datetime.timezone.utc)
        
        # 1. Account Age Score
        account_age_days = (current_time - user_data.created_at).days
        age_score = min(account_age_days / 365, 1.0)  # Normalize to 0-1 (1 year = 1)
        logger.debug(f"Account age: {account_age_days} days, Score: {age_score:.2f}")
        
        # 2. Follower/Following Analysis
        metrics = user_data.public_metrics
        followers = metrics.get("followers_count", 0)
        following = metrics.get("following_count", 0)
        tweet_count = metrics.get("tweet_count", 0)
        listed_count = metrics.get("listed_count", 0)
        
        # Calculate follower/following ratio with smoothing
        ratio = followers / (following + 1)  # Add 1 to avoid division by zero
        ratio_score = min(ratio / 10, 1.0)  # Cap at 10
        logger.debug(f"Follower ratio: {ratio:.2f}, Score: {ratio_score:.2f}")
        
        # 3. Bio and Profile Analysis
        bio = user_data.description or ""
        bio_score = 0.5  # Base score
        
        # Bio length bonus (10-100 chars is ideal)
        bio_length = len(bio)
        if 10 <= bio_length <= 100:
            bio_score += 0.2
        
        # Profile completeness
        profile_complete = 0
        if bio:
            profile_complete += 1
        if user_data.profile_image_url and "default_profile" not in user_data.profile_image_url:
            profile_complete += 1
        if user_data.location:
            profile_complete += 1
        if user_data.url:
            profile_complete += 1
            
        bio_score += (profile_complete / 4) * 0.3  # Up to 0.3 points for complete profile
        bio_score = min(bio_score, 1.0)
        logger.debug(f"Bio score: {bio_score:.2f}")
        
        # 4. Tweet Engagement Analysis
        engagement_score = 0
        avg_likes = 0
        avg_retweets = 0
        
        try:
            # Get recent tweets (last 10)
            tweets = client.get_users_tweets(
                id=user_id,
                max_results=10,
                tweet_fields=["public_metrics", "created_at"],
                exclude=["retweets", "replies"]
            )
            
            if tweets.data:
                tweet_metrics = [t.public_metrics for t in tweets.data]
                avg_likes = sum(t.get("like_count", 0) for t in tweet_metrics) / len(tweet_metrics)
                avg_retweets = sum(t.get("retweet_count", 0) for t in tweet_metrics) / len(tweet_metrics)
                
                # Normalize engagement (log scale as engagement follows power law)
                engagement_score = min(
                    (np.log1p(avg_likes) + np.log1p(avg_retweets * 2)) / 8,
                    1.0
                )
                
                logger.debug(f"Avg likes: {avg_likes:.1f}, Avg RTs: {avg_retweets:.1f}, Engagement score: {engagement_score:.2f}")
                
        except Exception as e:
            logger.warning(f"Error analyzing tweets for user {user_id}: {e}")
        
        # 5. Sentiment Analysis
        sentiment_score = 0.5  # Neutral default
        if tweets and tweets.data:
            try:
                sentiments = []
                for tweet in tweets.data:
                    # Skip very short tweets for sentiment analysis
                    if len(tweet.text) < 10:
                        continue
                        
                    # Analyze sentiment
                    analysis = TextBlob(tweet.text)
                    # Weight by tweet recency (more recent tweets have more weight)
                    days_old = (current_time - tweet.created_at).days
                    weight = max(1 - (days_old / 30), 0)  # Linear decay over 30 days
                    
                    # Combine polarity (-1 to 1) and subjectivity (0 to 1)
                    # We want more objective tweets to have higher weight
                    combined_score = (analysis.sentiment.polarity + 1) * (1 - analysis.sentiment.subjectivity)
                    sentiments.append((combined_score, weight))
                
                if sentiments:
                    # Calculate weighted average sentiment
                    total_weight = sum(w for _, w in sentiments)
                    if total_weight > 0:
                        weighted_sum = sum(s * w for s, w in sentiments)
                        sentiment_score = (weighted_sum / total_weight + 1) / 2  # Normalize to 0-1
                        logger.debug(f"Sentiment score: {sentiment_score:.2f}")
                        
            except Exception as e:
                logger.warning(f"Error in sentiment analysis: {e}")

        # 6. Trusted Accounts Verification
        trusted_score = 0.0
        trusted_followers = 0
        
        try:
            trusted_accounts = get_trusted_accounts()
            logger.info(f"Checking against {len(trusted_accounts)} trusted accounts")
            
            # Convert usernames to user IDs first for batch lookup
            trusted_users = []
            for username in trusted_accounts[:100]:  # Limit to 100 to avoid rate limits
                try:
                    user = client.get_user(username=username.strip())
                    if user and user.data:
                        trusted_users.append(user.data.id)
                except Exception as e:
                    logger.warning(f"Error looking up trusted user @{username}: {e}")
            
            # Check follows in batches
            for i in range(0, len(trusted_users), 100):
                batch = trusted_users[i:i+100]
                try:
                    relationships = client.get_users_following(
                        id=user_id,
                        max_results=100,
                        user_fields=["id"]
                    )
                    
                    if relationships and relationships.data:
                        followed_ids = {user.id for user in relationships.data}
                        trusted_followers += len(set(batch) & followed_ids)
                        
                        # Early exit if we already have enough trusted followers
                        if trusted_followers >= 3:
                            break
                            
                except Exception as e:
                    logger.warning(f"Error checking trusted follows batch: {e}")
            
            # Calculate score based on number of trusted followers
            if trusted_followers >= 3:
                trusted_score = 1.0
            elif trusted_followers == 2:
                trusted_score = 0.75
            elif trusted_followers == 1:
                trusted_score = 0.5
                
            logger.info(f"Found {trusted_followers} trusted followers, Score: {trusted_score:.2f}")
            
        except Exception as e:
            logger.error(f"Error in trusted accounts verification: {e}")
            trusted_score = 0.1  # Default low score if verification fails

        # 7. Calculate overall trustworthiness score
        weights = {
            "age": 0.15,         # Account age
            "ratio": 0.15,       # Follower/Following ratio
            "bio": 0.15,         # Profile completeness
            "engagement": 0.2,   # Tweet engagement
            "sentiment": 0.1,    # Tweet sentiment
            "trusted": 0.25,     # Trusted connections
        }
        
        # Apply weights to individual scores
        weighted_scores = {
            "age": age_score * weights["age"],
            "ratio": ratio_score * weights["ratio"],
            "bio": bio_score * weights["bio"],
            "engagement": engagement_score * weights["engagement"],
            "sentiment": sentiment_score * weights["sentiment"],
            "trusted": trusted_score * weights["trusted"]
        }
        
        # Calculate final score (0-100)
        overall_score = sum(weighted_scores.values()) * 100
        overall_score = min(max(overall_score, 0), 100)  # Ensure within bounds
        
        # Determine trust level
        if overall_score >= 80:
            trust_level = "🟢 High Trust"
        elif overall_score >= 50:
            trust_level = "🟡 Medium Trust"
        else:
            trust_level = "🔴 Low Trust"
        
        # Prepare detailed analysis
        analysis = {
            "user_id": user_id,
            "username": user_data.username,
            "name": user_data.name,
            "verified": user_data.verified,
            "account_created": user_data.created_at.strftime("%Y-%m-%d"),
            "account_age_days": account_age_days,
            "followers": followers,
            "following": following,
            "follower_ratio": round(ratio, 2),
            "tweet_count": tweet_count,
            "listed_count": listed_count,
            "bio": bio[:200] + ("..." if len(bio) > 200 else ""),
            "location": user_data.location or "Not specified",
            "url": user_data.url or "Not provided",
            "avg_likes": round(avg_likes, 1),
            "avg_retweets": round(avg_retweets, 1),
            "trusted_followers": trusted_followers,
            "trust_score": round(overall_score, 1),
            "trust_level": trust_level,
            "analysis_timestamp": current_time.isoformat(),
            "scores": {
                "raw": {
                    "age_score": age_score,
                    "ratio_score": ratio_score,
                    "bio_score": bio_score,
                    "engagement_score": engagement_score,
                    "sentiment_score": sentiment_score,
                    "trusted_score": trusted_score,
                },
                "weighted": weighted_scores
            }
        }
        
        logger.info(f"Analysis complete for @{user_data.username}. Score: {overall_score:.1f} ({trust_level})")
        return analysis
        
    except Exception as e:
        logger.error(f"Error in analyze_user: {e}", exc_info=True)
        return None
    except Exception as e:
        print(f"Error analyzing user {user_id}: {e}")
        return None

def generate_report(analysis: Dict[str, Any]) -> str:
    """
    Generate a human-readable trustworthiness report from analysis data.
    
    Args:
        analysis: Dictionary containing user analysis data
        
    Returns:
        Formatted report string
    """
    if not analysis:
        return "❌ Unable to analyze user. The account may be private or suspended."
    
    try:
        # Format verification badge
        verified_badge = "✅" if analysis.get("verified") else ""
        
        # Format account age
        years = analysis['account_age_days'] // 365
        months = (analysis['account_age_days'] % 365) // 30
        age_str = f"{years}y {months}m" if years > 0 else f"{months}m"
        
        # Format engagement metrics
        engagement = analysis['avg_likes'] + analysis['avg_retweets']
        
        # Generate report sections
        header = f"🔍 Trust Analysis for @{analysis['username']} {verified_badge}\n"
        header += f"🏷️ {analysis['name']}\n"
        header += f"🏆 {analysis['trust_level']} ({analysis['trust_score']}/100)\n\n"
        
        metrics = [
            f"👤 {analysis['followers']:,} followers | {analysis['following']:,} following | 📊 {analysis['follower_ratio']:.1f} ratio",
            f"📅 Account age: {age_str} | 🐦 {analysis['tweet_count']:,} tweets | 📌 {analysis['listed_count']:,} listed",
            f"💬 Avg engagement: {engagement:.1f} (👍 {analysis['avg_likes']:.1f} | 🔄 {analysis['avg_retweets']:.1f})",
            f"🤝 Trusted connections: {analysis['trusted_followers']}"
        ]
        
        # Add location and URL if available
        if analysis['location'] != "Not specified":
            metrics.append(f"📍 {analysis['location']}")
        if analysis['url'] != "Not provided":
            metrics.append(f"🔗 {analysis['url']}")
        
        # Add bio if present
        if analysis['bio']:
            metrics.append(f"\n📝 {analysis['bio']}")
        
        # Add disclaimer
        footer = "\n\n⚠️ This is an automated analysis. Always do your own research."
        
        # Combine all sections
        report = header + "\n".join(metrics) + footer
        
        # Ensure the report fits in a single tweet (280 chars)
        max_length = 280 - len(footer) - 10  # Leave room for ellipsis
        if len(report) > max_length:
            report = report[:max_length] + "..." + footer
            
        return report
        
    except Exception as e:
        logger.error(f"Error generating report: {e}", exc_info=True)
        return "❌ Error generating report. Please try again later."

class RugGuardStream(tweepy.StreamingClient):
    """Streaming client to monitor for mentions and trigger analysis."""
    
    def __init__(self, bearer_token, **kwargs):
        super().__init__(bearer_token, **kwargs)
        self.logger = logging.getLogger('rugguard.stream')
    
    def on_connect(self):
        """Called when the stream is connected."""
        self.logger.info("Successfully connected to Twitter Streaming API")
    
    def on_disconnect(self):
        """Called when the stream disconnects."""
        self.logger.warning("Disconnected from Twitter Streaming API")
    
    def on_errors(self, errors):
        """Handle any errors from the stream."""
        self.logger.error(f"Stream error: {errors}")
        return True  # Keep the stream alive
    
    def on_tweet(self, tweet):
        """Process incoming tweets that match our filter."""
        try:
            # Check if this is a reply with our trigger phrase
            if not (tweet.referenced_tweets and 
                   any(ref.type == "replied_to" for ref in tweet.referenced_tweets) and
                   "riddle me this" in tweet.text.lower()):
                return
            
            self.logger.info(f"Processing mention from tweet ID: {tweet.id}")
            
            # Check if we've already processed this tweet
            cursor.execute(
                "SELECT tweet_id FROM processed_tweets WHERE tweet_id = ?", 
                (str(tweet.id),)
            )
            if cursor.fetchone():
                self.logger.info(f"Skipping already processed tweet: {tweet.id}")
                return
            
            # Mark as processing
            cursor.execute(
                "INSERT INTO processed_tweets (tweet_id, status) VALUES (?, ?)",
                (str(tweet.id), "processing")
            )
            conn.commit()
            
            # Find the original tweet being replied to
            original_tweet_id = next(
                ref.id for ref in tweet.referenced_tweets 
                if ref.type == "replied_to"
            )
            
            # Get the original tweet to find the author
            original_tweet = client.get_tweet(
                original_tweet_id,
                tweet_fields=["author_id", "conversation_id"],
                user_auth=True
            )
            
            if not original_tweet or not original_tweet.data:
                self.logger.error(f"Could not find original tweet: {original_tweet_id}")
                self._update_tweet_status(tweet.id, "error:original_tweet_not_found")
                return
            
            # Analyze the original tweet's author
            analysis = analyze_user(original_tweet.data.author_id)
            if not analysis:
                self.logger.error("Failed to analyze user")
                self._update_tweet_status(tweet.id, "error:analysis_failed")
                return
            
            # Generate and post the report
            report = generate_report(analysis)
            
            try:
                response = client.create_tweet(
                    text=report,
                    in_reply_to_tweet_id=tweet.id,
                    user_auth=True
                )
                
                if response and response.data:
                    self.logger.info(f"Posted analysis for @{analysis['username']}")
                    self._update_tweet_status(tweet.id, "completed")
                    
                    # Log the analysis in history
                    cursor.execute(
                        """
                        INSERT INTO analysis_history 
                        (user_id, username, trust_score)
                        VALUES (?, ?, ?)
                        """,
                        (
                            analysis['user_id'],
                            analysis['username'],
                            analysis['trust_score']
                        )
                    )
                    conn.commit()
                else:
                    self.logger.error("Failed to post analysis")
                    self._update_tweet_status(tweet.id, "error:post_failed")
                    
            except Exception as e:
                self.logger.error(f"Error posting tweet: {e}", exc_info=True)
                self._update_tweet_status(tweet.id, f"error:post_exception:{str(e)[:50]}")
                
        except Exception as e:
            self.logger.error(f"Error in on_tweet: {e}", exc_info=True)
            try:
                self._update_tweet_status(tweet.id, f"error:unhandled:{str(e)[:50]}")
            except:
                pass
    
    def _update_tweet_status(self, tweet_id: str, status: str):
        """Update the status of a processed tweet."""
        try:
            cursor.execute(
                "UPDATE processed_tweets SET status = ? WHERE tweet_id = ?",
                (status, str(tweet_id))
            )
            conn.commit()
        except Exception as e:
            self.logger.error(f"Error updating tweet status: {e}", exc_info=True)

def setup_stream_rules(stream: RugGuardStream) -> bool:
    """Set up the streaming rules for the bot."""
    try:
        # Delete any existing rules
        rules = stream.get_rules()
        if rules.data:
            rule_ids = [rule.id for rule in rules.data]
            stream.delete_rules(rule_ids)
        
        # Add our rule to track mentions
        rule = tweepy.StreamRule(
            value="@projectrugguard",
            tag="rugguard_mentions"
        )
        stream.add_rules(rule)
        logger.info("Successfully set up streaming rules")
        return True
    except Exception as e:
        logger.error(f"Error setting up stream rules: {e}")
        return False

def cleanup():
    """Clean up resources before exiting."""
    try:
        if 'conn' in globals() and conn:
            conn.close()
        logger.info("Cleanup completed")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def signal_handler(signum, frame):
    """Handle signals for graceful shutdown."""
    logger.info(f"Received signal {signum}, shutting down...")
    cleanup()
    sys.exit(0)

# Main function to start the bot
def main():
    """Main entry point for the RUGGUARD bot."""
    logger.info("Starting RUGGUARD bot...")
    
    # Set up signal handlers for graceful shutdown
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize and configure the stream
        stream = RugGuardStream(
            bearer_token=BEARER_TOKEN,
            wait_on_rate_limit=True,
            max_retries=3
        )
        
        # Set up streaming rules
        if not setup_stream_rules(stream):
            logger.error("Failed to set up stream rules. Exiting.")
            return
        
        # Start streaming
        logger.info("Starting stream...")
        stream.filter(
            tweet_fields=[
                "author_id",
                "conversation_id",
                "created_at",
                "in_reply_to_user_id",
                "referenced_tweets",
                "text"
            ],
            expansions=[
                "author_id",
                "referenced_tweets.id",
                "in_reply_to_user_id"
            ],
            user_fields=[
                "created_at",
                "description",
                "location",
                "name",
                "profile_image_url",
                "protected",
                "public_metrics",
                "url",
                "username",
                "verified"
            ],
            user_auth=True
        )
        
    except Exception as e:
        logger.critical(f"Fatal error in main: {e}", exc_info=True)
        raise
    finally:
        cleanup()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
    finally:
        cleanup()
