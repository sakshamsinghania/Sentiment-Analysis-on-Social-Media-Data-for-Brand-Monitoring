# src/collect_tweets.py
"""
Usage:
    python src/collect_tweets.py "Nike" 2025-01-01 2025-01-07 1000

This will save tweets to data/Nike_2025-01-01_2025-01-07.csv
Requires: snscrape
"""
import sys
import os
from datetime import datetime
import pandas as pd

def collect_tweets(query, since=None, until=None, max_items=1000):
    import snscrape.modules.twitter as sntwitter
    q = query
    if since:
        q += f" since:{since}"
    if until:
        q += f" until:{until}"
    tweets = []
    for i, t in enumerate(sntwitter.TwitterSearchScraper(q).get_items()):
        if i >= max_items:
            break
        tweets.append({
            "date": t.date.isoformat(),
            "username": t.user.username,
            "content": t.content,
            "id": t.id,
            "url": f"https://twitter.com/{t.user.username}/status/{t.id}"
        })
    return pd.DataFrame(tweets)

def main():
    if len(sys.argv) < 4:
        print("Usage: python src/collect_tweets.py <brand-query> <since YYYY-MM-DD> <until YYYY-MM-DD> [max_items]")
        print("Example: python src/collect_tweets.py Nike 2025-01-01 2025-01-07 1000")
        sys.exit(1)
    query = sys.argv[1]
    since = sys.argv[2]
    until = sys.argv[3]
    max_items = int(sys.argv[4]) if len(sys.argv) >= 5 else 1000

    os.makedirs("data", exist_ok=True)
    df = collect_tweets(query, since, until, max_items=max_items)
    fname = f"data/{query.replace(' ','_')}_{since}_{until}.csv"
    df.to_csv(fname, index=False)
    print(f"Saved {len(df)} tweets to {fname}")

if __name__ == "__main__":
    main()
