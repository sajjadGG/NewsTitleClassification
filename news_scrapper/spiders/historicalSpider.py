from unicodedata import category
import scrapy 
from news_scrapper.items import ContentItem, TitleItem
from news_scrapper.settings import ITEM_OUTPUT_PATH
import json
import re

class KhabarOnlineScraper(scrapy.Spider):
    name = 'khabaronline'
    topic_to_id = {6:'Sport'}
    page_limit=200
    def start_requests(self):
        urls = [
            f"https://www.khabaronline.ir/archive?pi=1&tp={i}" for i in range(10000)
        ]
        for url in urls:
            # print(f"url is dsadsadas {url}")
            yield scrapy.Request(url=url,callback=self.parse)


    def parse(self, response):
        for item in response.css('section[id="box202"]').css('ul')[0].css('li'):
            title = item.css('h3').css('a::text')[0].get()
            category = item.css('p').css('span').css('a::text').get()
            yield TitleItem(title=title, category=category)
           
        try:
            if int(re.search('pi=(\d+)',response.url).groups(1))<self.page_limit:
                yield scrapy.Request(re.sub('pi=(\d+)', lambda exp: "pi={}".format(int(exp.groups()[0]) + 1), response.url), self.parse)
        except:
            pass