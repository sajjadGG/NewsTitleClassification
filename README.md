# Scraper:

Web Scraper to fetch data from the web to be used for a news title classification project.
There are specific spiders for each website plus a general xml news feed parser. First top news in each website is gathered and are passed to our pipeline to clean and write data in json format

## Quick Set up:

create a virtualenv
run `pip install -r requirements.txt`
run `main.py` in src with your desired spider to fetch most recent data
or you can use the published docker version
