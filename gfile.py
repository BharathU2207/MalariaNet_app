import gdown
def download_model():
    url = 'https://drive.google.com/file/d/1DSxmHveqSTQi5f1IU5qVu_DFARnzSZnc/view?usp=sharing' 
    file_id = url.split('/')[-2]
    prefix = 'https://drive.google.com/uc?/export=download&id='
    gdown.download(prefix+file_id)
