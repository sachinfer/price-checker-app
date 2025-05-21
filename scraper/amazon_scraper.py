import requests
from bs4 import BeautifulSoup
import re

def search_amazon(query):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    search_url = f"https://www.amazon.com/s?k={query.replace(' ', '+')}"
    res = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(res.content, "html.parser")

    products = []
    for item in soup.select(".s-result-item")[:3]:  # Top 3 results
        title = item.select_one("h2 a span")
        price = item.select_one(".a-price .a-offscreen")
        link = item.select_one("h2 a")

        if title and price and link:
            products.append({
                "store": "Amazon",
                "title": title.text,
                "price": price.text,
                "url": "https://www.amazon.com" + link['href']
            })

    return products
