# Scraper:

Web Scraper to fetch data from the web to be used for a news title classification project.
There are specific spiders for each website plus a general xml news feed parser. First top news in each website is gathered and are passed to our pipeline to clean and write data in json format

## Quick Set up:

create a virtualenv
run `pip install -r requirements.txt`
run `main.py` in src with your desired spider to fetch most recent data
or you can use the published docker version

# Preprocess

## Quick Set up:

to run preprocess you need to specify your desired configs in `configs/preprocess.ini` such as input file path
spacy model , etc.
to use desires spacy model such as `en_core_web_sm` you need to install them first using commands such as

`python -m spacy download en_core_web_sm`

then `cd src` and run

`python preprocess.py`
