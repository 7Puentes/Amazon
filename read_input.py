import libarchive.public


if __name__ == '__main__':

    train_file = '/Users/Charly/Downloads/train-tif-sample.zip'
    for entry in libarchive.public.file_pour(train_file):
        print(e)
