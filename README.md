# 🛡️ RUGGUARD - Solana Project Trust Analyzer

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fprojectrugguard)](https://twitter.com/projectrugguard)

> **Warning**  
> This project is currently in active development. Use at your own risk.

RUGGUARD is an intelligent Twitter bot that analyzes the trustworthiness of Solana project accounts. When someone replies to a tweet with "@projectrugguard riddle me this", the bot analyzes the original tweet's author and posts a trustworthiness report.

## 🌟 Features

- **Comprehensive Analysis**: Evaluates accounts based on multiple trust indicators
- **Real-time Monitoring**: Listens for mentions and responds automatically
- **Detailed Reports**: Provides clear, actionable insights about account trustworthiness
- **Open Source**: Transparent and community-driven development
- **Easy Deployment**: Ready to deploy on Replit or your own server
- Analyzes account metrics including:
  - Account age and activity
  - Follower/Following ratio
  - Bio content analysis
  - Engagement patterns
  - Sentiment analysis of recent tweets
  - Trusted account verification
- Posts concise trustworthiness reports
- Built-in rate limiting and duplicate detection

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Twitter Developer Account with Elevated Access
- Replit account (for deployment)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rugguard-bot.git
   cd ruguard-bot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root with the following variables:
   ```
   API_KEY=your_twitter_api_key
   API_SECRET=your_twitter_api_secret
   ACCESS_TOKEN=your_twitter_access_token
   ACCESS_TOKEN_SECRET=your_twitter_access_token_secret
   BEARER_TOKEN=your_twitter_bearer_token
   ```

## 🛠 Usage

1. Run the bot:
   ```bash
   python main.py
   ```

2. To test the bot, reply to any tweet with:
   ```
   @projectrugguard riddle me this
   ```

## 📊 How It Works

1. The bot monitors Twitter for mentions of "@projectrugguard riddle me this" in replies
2. When triggered, it identifies the original tweet's author
3. It analyzes various trust indicators:
   - Account age and activity
   - Follower/Following ratio
   - Bio content
   - Engagement metrics
   - Tweet sentiment
   - Trusted account verification
4. Generates a trust score (0-100)
5. Posts a reply with the analysis

## 🔧 Configuration

You can adjust the weights for the trust score calculation in `main.py` by modifying the `weights` dictionary in the `analyze_user` function.

## 🚀 Deployment to Replit

1. Fork this repository
2. Create a new Repl and import your forked repository
3. Set up environment variables in the Replit Secrets tab
4. Run the Repl

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
