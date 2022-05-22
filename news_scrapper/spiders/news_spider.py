import scrapy 
from news_scrapper.items import ContentItem
from news_scrapper.settings import ITEM_OUTPUT_PATH
import json
class NewsSpider(scrapy.Spider):
    name='news'

    def start_requests(self):
        urls = [
            'https://techcrunch.com/'
        ]
        for url in urls:
            yield scrapy.Request(url=url,callback=self.parse_content)
        
    def parse_content(self , response):

        for a in response.css('.post-block'):
            title = a.css('.post-block__header a.post-block__title__link::text').get()
            url = a.css('.post-block__header a.post-block__title__link').attrib['href']
            image_url = a.css('.post-block__media img').attrib['src']
            description = a.css('.post-block__content::text').get()
            yield ContentItem(title=title , url = url , image_url=image_url , description = description)
