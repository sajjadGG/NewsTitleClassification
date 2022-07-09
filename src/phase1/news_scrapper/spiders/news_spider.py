import traceback
import scrapy 
from news_scrapper.items import ContentItem,TitleItem
from news_scrapper.settings import ITEM_OUTPUT_PATH
import json
import re
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


class HuffingPostSpider(scrapy.Spider): 
    name = "HuffingPostSpider" 
    page_limit=200
    page_number=1
    # start_urls=[
    #         "https://www.huffpost.com/entertainment/?page=1", 
    #         "https://www.huffpost.com/news/world-news/?page=1", 
    #         "https://www.huffpost.com/impact/business?page=1" ,
            
    # ]
    def start_requests(self):
        urls=[
            "https://www.huffpost.com/entertainment?page=2", 
            "https://www.huffpost.com/news/world-news?page=2", 
            "https://www.huffpost.com/impact/business?page=2" ,
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)


    def parse(self, response): 
        for item in response.css('a[class="card__image__link"]').xpath("@aria-label"):
            print("byeeeeeeee"+response.url)
            category =  response.css('h2[class="zone__title__text"]')[0].css('*::text')[0].get()
            title = item.extract()
 
            yield TitleItem(title=title, category=category)
        try: 
            if int(re.search("page=(\d+)", response.url).groups(1)[0]) < self.page_limit: 
                print("hiiiiiiiiiiiiiiiiiiiii")
                yield scrapy.Request( 
                    re.sub( 
                        "page=(\d+)", 
                        lambda exp: "page={}".format(int(exp.groups()[0]) + 1), 
                        response.url, 
                    ), 
                    self.parse, 
                ) 
        except Exception as e : 
            traceback.print_exc()