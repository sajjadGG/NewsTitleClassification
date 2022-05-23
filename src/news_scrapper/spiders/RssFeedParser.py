from unicodedata import category
import scrapy
from scrapy.spiders import XMLFeedSpider
from news_scrapper.items import ContentItem


class RssfeedparserSpider(XMLFeedSpider):
    name = 'RssFeedParser'
    # allowed_domains = ['cnn.com']
    start_urls = ['https://rss.nytimes.com/services/xml/rss/nyt/Education.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/Business.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/MediaandAdvertising.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml',
    'https://www.nytimes.com/services/xml/rss/nyt/Health.xml',
    ]
    itertag='item'

    def parse_node(self, response , node):
        category=response.url.split('/')[-1][:-4]
        title = node.xpath('//title/text()').get() 
        url = node.xpath('//link/text()').get()
        description = node.xpath('//description/text()').get()

        yield ContentItem(title=title , url = url , description = description,category=category)