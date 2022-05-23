from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from news_scrapper.spiders.news_spider import NewsSpider
from news_scrapper.spiders.RssFeedParser import RssfeedparserSpider
from news_scrapper.spiders.historicalSpider import KhabarOnlineScraper
from news_scrapper.spiders.news_spider import HuffingPostSpider

process = CrawlerProcess(get_project_settings())
process.crawl(RssfeedparserSpider)
process.start()
