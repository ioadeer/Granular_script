#!/usr/bin/env python3

""" NOTA: bastante parte del script hace analisis de PCA y termina produciendo
data frames. quizas tenga mas sentido hacer eso en un script dedicado al
analisis y procesamiento y tener otro script que solo se dedique a importar el
csv y el audio y tocarlos.

hacer un script que reciba un wav y csv y que reproduzca un audio según el
reordenamiento de los frames con distintos criterios: según valores de PCA o de
features.  en una primera instancia, el tamaño de chunk va a estar supeditado
al tamaño de ventana de análisis utilizado durante la etapa de análisis. hay
que ver qué camino tomar: durante la etapa de análisis se puede plantear
realizar la medición de los features con distinto tamaño de ventana de análisis
y guardar los outputs como distintos data frames para después tener variedad de
tamaño de ventana de análisis al momento de la concatenación.  """

import numpy as np
import wave #as wv # antes veni usando scipy.io.wave me paso a este porque está en ejemplo de pyaudio
import scipy.io.wavfile as wf
import pyaudio
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import getopt
import sys
import os
sys.path.append('..')
from utils import granular_util as gu

def usage():
    print(" Script que recibe como input un archivo wav, su info en csv y lo reproduce")
    print("Example: python -f <wav file name> -c <csv file name>")
    print("Example: python --wavfile <wav file name> --csvfile <csv file name>")

def main():
    try:
        (opts, args) = getopt.getopt(sys.argv[1:], 'c:f:h',['help','wavfile','csvfile'])  # 'f:h' tienen que estar en orden alfabetico
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)

    filepath = None
    frameSize = None
    hopSize = None

    if len(opts) != 0:
        for (o,a) in opts:
            if o in ('-h', '--help'):
                usage()
                sys.exit(2)
            elif o in ('-f','--wavfile'):
                filepath = a
            elif o in ('-c','--csvfile'):
                csvfilepath= a
            else:
                usage()
                sys.exit(2),
    else:
        # no options passed
        usage()
        sys.exit(2)

    if csvfilepath and filepath:
        # parse csvfilepath to retrieve frameSize and hopSize values
        tempFileList = csvfilepath.split('.')[0].split('_')
        frameSize = int(tempFileList[tempFileList.index("frameSize")+1])
        hopSize = int(tempFileList[tempFileList.index("hopSize")+1])
        #import csv file
        df = pd.read_csv(csvfilepath)
        # sklearn
        # select columns from dataframe in order to obtain PCA values faster
        # en esta seccion estaria bien variar la eleccion de features y hacerla interactiva
        dfs = []
        dfs.append(df.filter(regex='spectral_contrast_coeffs_\d'))
        dfs.append(df.filter(regex='mfcc_\d'))
        dfs.append(df.filter(items=['spectral_centroid', 'pitch_salience', 'spectral_energy']))
        selectionDf = pd.concat(dfs, axis=1) 
        # x va a ser directamente un data frame con datos estandarizados
        # ver si tiene sentido estandarizar selectionDf tambien
        x = StandardScaler().fit_transform(selectionDf)
        pca = PCA(n_components =2)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data = principalComponents,
                                    columns = ['principal component 1',
                                        'principal component 2'])
        tempDfs = []
        tempDfs.append(principalDf)
        tempDfs.append(selectionDf)
        componentSelectionDf = pd.concat(tempDfs, axis=1)
        # obtener lista para reordenar
        principal_component_1 = np.array(principalDf['principal component 1'].values)
        by_pc1_sorted = np.argsort(principal_component_1)
        #array_to_sort = np.array(selectionDf['spectral_centroid'].values) # pitch_salience, spectral_centroid, spectral_energy
        #by_feature_sorted = np.argsort(array_to_sort)
        #abrir archivo
        audio_file = wave.open(filepath, 'rb') 
        audio_bytes= audio_file.readframes(audio_file.getnframes())  
        #audio_data = np.frombuffer(audio_bytes)
        #audio_data.dtype = np.float32
        rate, audio_data = wf.read(filepath) 
        # re arranged data
        rearranged = gu.rearrange(frameSize, hopSize, audio_data, by_pc1_sorted, 'hanning')
        #rearranged = gu.rearrange(frameSize, hopSize, audio_data, by_feature_sorted, 'hanning')
        rearranged_output = rearranged.astype(np.int16)
        rearranged_audio_bytes = rearranged_output.tobytes()
        # instantiate PyAudio lo voy a hacer sin callback en el primer intento
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(audio_file.getsampwidth()), # 1 mono 2 stereo
                        channels = audio_file.getnchannels(),
                        rate = audio_file.getframerate(),
                        output = True)
        CHUNK = 64 #512 #1024
        START_POINT = 0
        output = rearranged_audio_bytes[START_POINT:CHUNK]
        
        # play stream (3)
        while len(output) > 0:
            stream.write(output)
            START_POINT += CHUNK
            output = rearranged_audio_bytes[START_POINT:START_POINT+CHUNK]
        
        # stop stream (4)
        stream.stop_stream()
        stream.close()
        
        # close PyAudio (5)
        p.terminate()

if __name__ == "__main__":
    main() 
