from .import prep_data
from .import transformacion
import argparse

__intro__ = """

   __         _ _     _                          _       _   _
  / _|       | | |   (_)                        | |     | | (_)
 | |_ ___  __| | |__  _  ___  _ __ ___   ___  __| |   __| |  _    __   ___  _ __ ___
 |  _/ _ \/ _` | '_ \| |/ _ \| '_ ` _ \ / _ \/ _` |  / _` | | |  / _| / _ \| '_  `_ \ 
 | ||  __/ (_| | |_) | | (_) | | | | | |  __/ (_| |-| (_| | | | | (_ | (_) | | | | | |
 |_| \___|\__,_|_.__/|_|\___/|_| |_| |_|\___|\__,_|  \__,_| |_|  \__| \ __/|_| |_| |_|


"""


def intro():
    """Prints intro for the CLI"""

def main(dcm_path,num_dicoms): 
    print("\033[91m" + __intro__ +"\033[0m")
    print("Inicio transformacion DICOM")
    transformacion.main(dcm_path,num_dicoms)
    print("Fin transformacion DICOM")
    print("Inicio preproceso datos")
    prep_data.main()
    print("Fin preproceso datos")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script de aprendizaje federado")
    parser.add_argument("--dcm_path",type=str,required=True, help="Ruta del dataset")
    parser.add_argument("--num_dicoms",type=int,required=True,help="Número de archivos dicom")
    args = parser.parse_args()
    main(args.dcm_path,args.num_dicoms)