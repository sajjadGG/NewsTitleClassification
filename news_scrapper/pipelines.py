# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from news_scrapper.settings import ITEM_OUTPUT_PATH , BATCH_SIZE , API_ENDPOINT
import requests
import json

class JsonWriterPipeline:

    def open_spider(self , spider):
        self.file = open(ITEM_OUTPUT_PATH , 'w',encoding='utf-8')

    def close_spider(self , spider):
        self.file.close()

    def process_item(self, item, spider):
        line = json.dumps(ItemAdapter(item).asdict(),ensure_ascii=False) + '\n'
        self.file.write(line)
        return item

class RequestPipeline:

    def open_spider(self , spider):
        self.counter = 0
        self.batch_size = BATCH_SIZE
        self.body = []

    def close_spider(self, spider):
        if self.counter > 0:
            self.send_request()
        self.counter = 0
        self.body = []

    def process_item(self , item , spider):
        self.update_body(item)
        self.counter+=1
        if self.counter > self.batch_size:
            self.send_request()
            self.body = []
            self.counter = 0
    #TODO : move to utils and make purification more efficient
    def purify_url(self , url):
        return url.split('//')[-1].split('/')[0] ,'/'+'/'.join(url.split('//')[-1].split('/')[1:])

    def update_body(self , item):
        item = ItemAdapter(item).asdict()
        domain , path = self.purify_url(item['url'])
        self.body+=[{'image' : item['image_url'] , 'domain' : domain , 'path' : path, 
                'des': item['description'] , 'title' : item['title']}]
        
    def send_request(self):
        # res = requests.post(API_ENDPOINT , json = self.body)
        # print(res)
        pass
