import os
import argparse
from pathlib import Path
import instaloader
import youtube_dl

def main(args):

  if args.source == 'instagram':

    # NEW
    # hashtags = [ 
    #   'handflower',
    #   'handsinthedirt',
    #   'handsinframe',
    #   'snacks',
    #   'plantbasedlife',
    #   ''
    #   ]

    #crazyshopping #foodshopping #forestfire #river #riverstream
    # hashtags = [ 
    #     'bugs', 
    #     'spring', 
    #     'wonderful_places', 
    #     'putrefaction', 
    #     'rottenfood', 
    #     'slaughterhouse', 
    #     'brokennail', 
    #     'pollutedair', 
    #     '4wd', 
    #     'meatshopping', 
    #     'plasticbags', 
    #     'plasticsurgery',
    #     'naturallight',
    #     'mining',
    #     'miningfarm',
    #     'environmentalactivism',
    #     'lappingwaves',
    #     'stopkm',
    #     'globalwarming',
    #     'riots',
    #     'environmentalprotection',
    #     'disposables',
    #     'arctic',
    #     'fish',
    #     'coastprotectors',
    #     'nopipelines',
    #     'nddl',
    #     'notredamedeslandes',
    #     'radicalecology',
    #     'barricade',
    #     'landdefense',
    #     'landdefenders',
    #     'zad',
    #     'seashepherd',
    #     'industrialfarming',
    #     'animalabuse',
    #     'animalcruelty'   
    #     ]

    hashtags = [ 
        'Deathvalley', 
        'Modernoutdoors', 
        'nature_hippys',
        'dumpsterdiving',
        'dumpster',
        'eruptions',
        'industrialphotography',
        'oilrefinery',
        'oilspill',
        'factoryworker',
        'assemblyline',
        'bitcoinmining',
        'financialmarkets',
        'ecoactivism',
        'miningindustry',
        'specism',
        'salmonfarm'
        ]

    
    loader = instaloader.Instaloader()

    # ▶ instaloader '#river' --post-filter="is_video" --no-video-thumbnails --no-captions --no-metadata-json --no-profile-pic --count xxx
    loader.download_video_thumbnails = False
    loader.download_comments = False
    loader.save_metadata = False
    loader.download_geotags = False
    loader.post_metadata_txt_pattern = ''

    # Downloading
    post_filter = lambda post: post.is_video
    for hashtag in hashtags:
      loader.dirname_pattern = os.path.join(args.destination, args.source, '#' + hashtag)
      posts = loader.download_hashtag(hashtag, max_count=args.count, post_filter=post_filter)

    # Cleaning up
    for p in Path(args.destination).glob('*/*.jpg'):
      p.unlink()

  elif args.source == 'youtube':

    search_terms = [
      # 'explosion+chemical+plant',
      # 'explosion+filmed+on+mobile',
      'forest+fire',
      'eco+riots',
      'tar+sands',
      'oil+spill',
      'slaughterhouse',
      'deforestation',
      'deforestation+footage',
      'palm+ oil+footages',
      'nestlé+water+scandal',
      'nestlé+water+documentary',
      'bhopal+disaster',
      'notre dame des landes',
      'acide camion arcelormittal'
    ]

    # ▶ youtube-dl --default-search "ytsearch20" "free+stock+footage+tar+moss" -vic -o '' --restrict-filenames -f bestvideo --max-filesize 10M --no-part
    # https://github.com/rg3/youtube-dl/blob/3e4cedf9e8cd3157df2457df7274d0c842421945/youtube_dl/YoutubeDL.py#L137-L312
    ydl_opts = {
      'outtmpl': os.path.join(args.destination, '%(id)s.%(ext)s'),
      'format': 'mp4',
      'default_search': 'ytsearch'+str(args.count),
      'restrictfilenames': True,      
      # 'max_filesize': 20000000,
      'nopart': True
    }
    for search_term in search_terms:
      ydl_opts['outtmpl'] = os.path.join(args.destination, args.source, search_term, '%(id)s.%(ext)s')
      with youtube_dl.YoutubeDL(ydl_opts) as ydl:
          ydl.download([search_term])


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='TBC')
  parser.add_argument('-c', '--count', type=int, help='Number of posts to download', default=50)
  parser.add_argument('-s', '--source', type=str, help='Source to scrap', default='instagram', choices=['instagram', 'youtube'])
  parser.add_argument('-d', '--destination', type=str, help='Destination folder')

  args = parser.parse_args()
  main(args) 