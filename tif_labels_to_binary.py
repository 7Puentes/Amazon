import gdal
import os
import numpy as np
import csv
import zipfile
import tarfile

#Path donde estan descargados los archivos train-tif-sample.tar y train.csv.zip
train_data = '/home/tat/Escritorio/TIP/Amazonas/Inputs/'

train_data_dir_tar = train_data + 'train-tif-sample.tar'
train_data_dir_untar = train_data
train_data_dir = train_data + "train-tif-sample/"

train_input_dir_zip = train_data + 'train.csv.zip'
train_input_dir_unzip = train_data
train_input_dir = train_data + 'train.csv'

binary_dir_files="binary_tifs"

def untar_file(path_to_tar, path_to_untar):
    tar_ref = tarfile.TarFile(path_to_tar, 'r')
    tar_ref.extractall(path_to_untar)
    tar_ref.close()

def zipdir(path_dir_to_zip, path_to_zip_file):
    zipf = zipfile.ZipFile(path_to_zip_file, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(path_dir_to_zip):
        for file in files:
            zipf.write(os.path.join(root, file))

def unzip_file(path_to_zip, path_to_unzip):
    zip_ref = zipfile.ZipFile(path_to_zip, 'r')
    zip_ref.extractall(path_to_unzip)
    zip_ref.close()


#*******************************************************************#
# Toma como parametro la direccion donde estan los archivos tif     #
# Retorna un diccionario clave-valor donde:                         #
#   * La clave es el nombre del archivos                            #
#   * El valor una lista de numeros (cada uno representa un tags)   #
#*******************************************************************#
def tif_with_tags(train_input_dir):
    inputReader = csv.reader(open(train_input_dir), delimiter=',', quotechar='|')
    rows_dict = {}
    values_dict = {}
    values_las_num = 0
    next(inputReader) # salteo primer linea
    for row in inputReader:
        labels = row[1].split(" ")
        labels_num = []
        for l in labels:
            if not l in values_dict:
                values_dict[l] = values_las_num
                values_las_num = values_las_num + 1
            labels_num.append(values_dict[l]) 
        rows_dict[row[0]]=labels_num

    return rows_dict


#*******************************************************************#
# Toma como input una lista de numeros del 0 al 16 que representan  #
# a los labels                                                      #
# Retorna dos bytes donde los bits encendidos son las posiciones    #
# de los tags                                                       #
#   Ej: [0, 3] son los labels                                       #
#       retorna 9 (00000000000000000000000000001001)                #                                       #
#*******************************************************************#
def get_binary_labels(labels):
    # necesitamos minimo 2 bytes porque son 17 tags posibles (se mepean del 0 al 16)
    binary_labels = int('00000000000000000000000000000000',2)
    for l in labels:
        binary_labels = binary_labels | (2**l) # or entre lo que ya tengo y si enciendo o no otro bit segun la posicion
    return binary_labels


#*******************************************************************#
# Crea los archivos binarios con datos de los archivos tif + labels # 
# Toma como input:                                                  #
#   train_data_dir: Path a la carpeta donde estan los archivos tif  #
#   rows_dict: Diccionario                                          #
#       * clave: el nombre del archivo                              #
#       * valor: lista de tags (numeros del 0 al 16) de la imagen   #
#   binary_dir_files: Nombre de la carpeta que se creara para       #
#   guardar los archivos binarios                                   #
#*******************************************************************#      
def save_tif_in_binary(train_data_dir, rows_dict, binary_dir_files):
    os.makedirs(binary_dir_files)

    count = 0
    for f in os.listdir(train_data_dir):
        if f.endswith('tif'):


            try:
                name = f.replace(".tif","")
                binary_name_file = binary_dir_files + "/" + name
                output = open(binary_name_file, "wb")
                ds = gdal.Open(train_data_dir + f)

                output.write(ds.GetRasterBand(1).ReadAsArray())
                output.write(ds.GetRasterBand(2).ReadAsArray())
                output.write(ds.GetRasterBand(3).ReadAsArray())
                output.write(ds.GetRasterBand(4).ReadAsArray())

                labels = rows_dict[name]  
                binary_labels = get_binary_labels(labels)              

                output.write(np.array(binary_labels,dtype=np.uint16))
                output.close()

                #print "Converted %s" % f
                count +=1
            except Exception as e:
                print e
    output.close()
    #print count


if __name__ == '__main__':

    #unzip_file(train_input_dir_zip, train_input_dir_unzip)
    #untar_file(train_data_dir_tar,train_data_dir_untar)
    #untar_file(train_data_dir_tar,train_data_dir_untar)
    train_data_dir = "/Users/Charly/Downloads/train-tif-sample/"
    binary_dir_files = "/Users/Charly/Downloads/train-bin-sample/"
    train_input_dir = "/Users/Charly/Downloads/train.csv"

    rows_dict = tif_with_tags(train_input_dir)
    save_tif_in_binary(train_data_dir, rows_dict,binary_dir_files)
    #zipdir(binary_dir_files,binary_dir_files+".zip" )