# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from unicodedata import category
import scrapy
from scrapy.item import Item, Field


class TitleItem(Item):
    title = Field()
    category = Field()
    
class ContentItem(Item):
    title = Field()
    description = Field()
    url = Field()
    image_url = Field()
    category = Field()
