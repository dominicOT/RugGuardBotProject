import tweepy
import time
import os
import sqlite3
import datetime
import requests
from textblob import TextBlob
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Twitter API credentials
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# Initialize Tweepy client
auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)
client = tweepy.Client(
    bearer_token=BEARER_TOKEN,
    consumer_key=API_KEY,
    consumer_secret=API_SECRET,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET,
    wait_on_rate_limit=True
)

# SQLite database setup
conn = sqlite3.connect("processed_tweets.db")
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS processed_tweets (tweet_id TEXT PRIMARY KEY)''')
conn.commit()

# Trusted accounts list from GitHub
TRUSTED_ACCOUNTS_URL = "https://raw.githubusercontent.com/devsyrem/turst-list/main/list"
def get_trusted_accounts():
    try:
        response = requests.get(TRUSTED_ACCOUNTS_URL)
        response.raise_for_status()
        return response.text.strip().split("\n")
    except requests.RequestException as e:
        print(f"Error fetching trusted accounts: {e}")
        return []

# Analyze user trustworthiness
def analyze_user(user_id):
    try:
        user = client.get_user(id=user_id, user_fields=["created_at", "description", "public_metrics"])
        if not user.data:
            return None

        # Account age
        account_age = (datetime.datetime.now(datetime.timezone.utc) - user.data.created_at).days
        age_score = min(account_age / 365, 1.0)  # Normalize to 0-1 (1 year = 1)

        # Follower/following ratio
        followers = user.data.public_metrics.get("followers_count", 0)
        following = user.data.public_metrics.get("following_count", 0)
        ratio = followers / (following + 1)  # Avoid division by zero
        ratio_score = min(ratio / 10, 1.0)  # Normalize, cap at 10

        # Bio content analysis
        bio = user.data.description or ""
        bio_score = 0.5
        if len(bio) > 10:
            bio_score += 0.2
        if any(keyword in bio.lower() for keyword in ["official", "verified", "expert"]):
            bio_score += 0.3
        bio_score = min(bio_score, 1.0)

        # Engagement patterns
        tweets = client.get_users_tweets(id=user_id, max_results=10, tweet_fields=["public_metrics"])
        engagement_score = 0
        if tweets.data:
            avg_likes = sum(t.public_metrics["like_count"] for t in tweets.data) / len(tweets.data)
            avg_retweets = sum(t.public_metrics["retweet_count"] for t in tweets.data) / len(tweets.data)
            engagement_score = min((avg_likes + avg_retweets) / 50, 1.0)  # Normalize, cap at 50

        # Sentiment analysis of recent tweets
        sentiment_score = 0
        if tweets.data:
            sentiments = [TextBlob(t.text).sentiment.polarity for t in tweets.data]
            avg_sentiment = sum(sentiments) / len(sentiments)
            sentiment_score = (avg_sentiment + 1) / 2  # Normalize -1 to 1 -> 0 to 1

        # Trusted accounts check
        trusted_accounts = get_trusted_accounts()
        trusted_followers = 0
        for trusted_id in trusted_accounts:
            try:
                trusted_id = trusted_id.strip()
                if trusted_id and client.get_friendship(source_id=user_id, target_id=trusted_id).following:
                    trusted_followers += 1
            except Exception as e:
                print(f"Error checking trusted account {trusted_id}: {e}")
        trusted_score = 1.0 if trusted_followers >= 2 else 0.5 if trusted_followers == 1 else 0.0

        # Calculate overall trustworthiness score
        weights = {
            "age": 0.2,
            "ratio": 0.2,
            "bio": 0.2,
            "engagement": 0.2,
            "sentiment": 0.1,
            "trusted": 0.3
        }
        overall_score = (
            age_score * weights["age"] +
            ratio_score * weights["ratio"] +
            bio_score * weights["bio"] +
            engagement_score * weights["engagement"] +
            sentiment_score * weights["sentiment"] +
            trusted_score * weights["trusted"]
        ) * 100  # Scale to 0-100

        return {
            "username": user.data.username,
            "account_age_days": account_age,
            "follower_ratio": round(ratio, 2),
            "bio": bio,
            "avg_engagement": round(avg_likes + avg_retweets, 2),
            "sentiment": round(avg_sentiment, 2),
            "trusted_followers": trusted_followers,
            "trust_score": round(overall_score, 2)
        }
    except Exception as e:
        print(f"Error analyzing user {user_id}: {e}")
        return None

# Generate trustworthiness report
def generate_report(analysis):
    if not analysis:
        return "Unable to analyze user due to an error or restricted access."
    report = (
        f"Trustworthiness Report for @{analysis['username']}:\n"
        f"- Account Age: {analysis['account_age_days']} days\n"
        f"- Follower/Following Ratio: {analysis['follower_ratio']}\n"
        f"- Bio: {'Present' if analysis['bio'] else 'Empty'}\n"
        f"- Avg Engagement (Likes+RTs): {analysis['avg_engagement']}\n"
        f"- Recent Tweet Sentiment: {'Positive' if analysis['sentiment'] > 0 else 'Neutral' if analysis['sentiment'] == 0 else 'Negative'}\n"
        f"- Trusted Followers: {analysis['trusted_followers']}/3\n"
        f"- Trust Score: {analysis['trust_score']}/100"
    )
    return report

# Stream listener for replies
class ReplyStream(tweepy.StreamingClient):
    def on_tweet(self, tweet):
        if tweet.referenced_tweets and tweet.text.lower().find("riddle me this") != -1:
            # Check if tweet is a reply
            for ref in tweet.referenced_tweets:
                if ref.type == "replied_to":
                    # Check if already processed
                    cursor.execute("SELECT tweet_id FROM processed_tweets WHERE tweet_id = ?", (tweet.id,))
                    if cursor.fetchone():
                        return
                    # Mark as processed
                    cursor.execute("INSERT INTO processed_tweets (tweet_id) VALUES (?)", (str(tweet.id),))
                    conn.commit()
                    try:
                        # Get the original tweet
                        original_tweet = client.get_tweet(ref.id, user_fields=["id"]).data
                        if not original_tweet:
                            return
                        # Analyze the original tweet's author
                        analysis = analyze_user(original_tweet.author_id)
                        if not analysis:
                            return
                        # Generate and post report
                        report = generate_report(analysis)
                        client.create_tweet(
                            text=report,
                            in_reply_to_tweet_id=tweet.id
                        )
                        print(f"Posted report for @{analysis['username']}")
                    except Exception as e:
                        print(f"Error processing tweet {tweet.id}: {e}")

# Main function to start the bot
def main():
    stream = ReplyStream(BEARER_TOKEN)
    stream.add_rules(tweepy.StreamRule("@projectrugguard"))
    stream.filter(tweet_fields=["referenced_tweets"], expansions=["referenced_tweets.id"])

if __name__ == "__main__":
    main()
