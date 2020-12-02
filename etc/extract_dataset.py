import pandas as pd

PATH_TO_A2D = './'

def extract():
    videoset_path = PATH_TO_A2D + 'videoset.csv'

    videoset = pd.read_csv(videoset_path)

    # Ball videos in the dataset will have ID of 34, 35, 36, 39
    # flying is 34, jumping is 35, rolling is 36, none is 39.

    # First get the names of all the ball videos.

    # Do this so we can more easily index the column
    videoset['Label_str'] = videoset['Label'].astype(str)

    # Only get ball videos
    videoset = videoset[videoset.Label_str.str.startswith('3')]

    # At this point all rows are referecning ball videos.
    # 253 are classified as training samples. 62 are classified as testing samples based on the provided usage.

    train_df = videoset[videoset['Usage'] == 0]
    test_df = videoset[videoset['Usage'] == 1]
    
    

extract()
