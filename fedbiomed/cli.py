# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import argparse
import importlib

from fedbiomed.common.constants import DEFAULT_NODE_NAME, DEFAULT_RESEARCHER_NAME
from fedbiomed.common.config import docker_special_case
from fedbiomed.common.cli import CLIArgumentParser, CommonCLI



class UniqueStore(argparse.Action):
    """Argparse action for avoiding having several time the same optional
      argument"""
    def __call__(self, parser, namespace, values, option_string):
        if getattr(namespace, self.dest, self.default) is not self.default:
            parser.error(option_string + " appears several times.")
        setattr(namespace, self.dest, values)


class ComponentParser(CLIArgumentParser):
    """Instantiates configuration parser"""

    def initialize(self):
        """Initializes argument parser for creating configuration file."""

        self._parser = self._subparser.add_parser(
            "component",
            help="The helper for generating or updating component configuration files, see `configuration -h`"
            " for more details",
        )

        self._parser.set_defaults(func=self.default)

        # Common parser to register common arguments for create and refresh
        common_parser = argparse.ArgumentParser(add_help=False)
        common_parser.add_argument(
            "-p",
            "--path",
            action=UniqueStore,
            metavar="COMPONENT_PATH",
            type=str,
            nargs="?",
            required=False,
            help="Path to specificy where Fed-BioMed component will be intialized.",
        )

        common_parser.add_argument(
            "-c",
            "--component",
            metavar="COMPONENT_TYPE[ NODE|RESEARCHER ]",
            type=str,
            nargs="?",
            required=True,
            help="Component type NODE or RESEARCHER",
        )

        # Create sub parser under `configuration` command
        component_sub_parsers = self._parser.add_subparsers()

        create = component_sub_parsers.add_parser(
            "create",
            parents=[common_parser],
            help="Creates component folder for the specified component if it does not exist. "
            "If the component folder exists, leave it unchanged",
        )

        create.add_argument(
            "-eo",
            "--exist-ok",
            action="store_true",
            help="Creates configuration only if there isn't an existing one",
        )

        create.set_defaults(func=self.create)

    def _get_component_instance(self, path: str, component: str):
        """Gets component"""
        if component.lower() == "node":
            config_node = importlib.import_module("fedbiomed.node.config")
            _component = config_node.node_component
        elif component.lower() == "researcher":
            os.environ["FBM_RESEARCHER_COMPONENT_ROOT"] = path
            config_researcher = importlib.import_module(
                "fedbiomed.researcher.config"
            )
            _component = config_researcher.researcher_component
        else:
            print(f"Undefined component type {component}")
            sys.exit(101)

        return _component

    def create(self, args):
        """CLI Handler for creating configuration file and assets for given component
        """
        if args.component is None:
            CommonCLI.error("Error: bad command line syntax")

        if not args.path:
            if args.component.lower() == "researcher":
                component_path = os.path.join(os.getcwd(), DEFAULT_RESEARCHER_NAME)
            else:
                component_path = os.path.join(os.getcwd(), DEFAULT_NODE_NAME)
        else:
            component_path = args.path

        # Researcher specific case ----------------------------------------------------
        # This is a special case since researcher import
        if args.component.lower() == "researcher":
            if DEFAULT_RESEARCHER_NAME in component_path and \
                os.path.isdir(component_path) and \
                not docker_special_case(component_path):
                if not args.exist_ok:
                    CommonCLI.error(
                        f"Default component is already existing. In the directory {component_path} "
                        "please remove existing one to re-initiate"
                    )
                else:
                    CommonCLI.success(
                        "Component is already existing. Using existing component."
                    )
                    return
            else:
                self._get_component_instance(component_path, args.component)
                return
        else:
            component = self._get_component_instance(component_path, args.component)
            # Overwrite force configuration file
            if component.is_component_existing(component_path):
                if not args.exist_ok:
                    CommonCLI.error(
                        f"Component is already existing in the directory `{component_path}`. To ignore "
                       "this error please execute component creation using `--exist-ok`"
                )
                else:
                    CommonCLI.success(
                        "Component is already existing. Using existing component."
                    )
                    return

            component.initiate(component_path)

        CommonCLI.success(f"Component has been initialized in {component_path}")



cli = CommonCLI()
cli.initialize_optional()

# Initialize configuration parser
configuration_parser = ComponentParser(cli.subparsers)
configuration_parser.initialize()

# Add node and researcher options
node_p = cli.subparsers.add_parser(
    "node", add_help=False, help="Command for managing Node component"
)
researcher_p = cli.subparsers.add_parser(
    "researcher", add_help=False, help="Command for managing Researcher component"
)

# Integracion con el modulo DICOM
def process_dicom(args):
    from fedbiomed.modulo_dicom import ejecutar_pipeline
    """
    Función que procesa los archivos DICOM ubicados en el directorio indicado por el argumento '--path'.
    Se asume que existe un módulo 'mi_modulo_dicom' con la función 'process_dicom_directory'.
    """
    try:
        ejecutar_pipeline.main(args.ruta_dcm, args.num_dcm, args.nom_nodo, args.clase_dg, args.metodo_bal)
    except ImportError:
        print("No se pudo encontrar el módulo de procesamiento DICOM")
        return
dicom_p = cli.subparsers.add_parser(
    "dicom", help="Procesa archivos DICOM y genera CSV a partir de ellos."
)
dicom_p.add_argument("--ruta_dcm",type=str,required=True, help="Ruta del dataset")
dicom_p.add_argument("--num_dcm",type=int,required=True,help="Número de archivos dicom")
dicom_p.add_argument("--nom_nodo",type=str,required=True,help="Ruta del nodo para la salida de los arhivos procesados")
dicom_p.add_argument("--clase_dg",type=int,required=True,choices=[1,0],help="Clase de diagnóstico: 0 (no cáncer), 1 (cáncer)")
dicom_p.add_argument("--metodo_bal",type=str,default="smote",choices=["smote", "smote-enn", "random-under"],help="Estrategia de balanceo (opcional).")
dicom_p.set_defaults(func=process_dicom)
# fin integracioncuad

# integracion balanceo
def process_dicom(args):
    from fedbiomed.preprocesador import preprocesamiento

    prep = preprocesamiento.PipelinePreprocesamiento()
    """
    Función que procesa los archivos DICOM ubicados en el directorio indicado por el argumento '--path'.
    Se asume que existe un módulo 'mi_modulo_dicom' con la función 'process_dicom_directory'.
    """
    try:
        prep.balanceo(args.ruta_csv, args.metodo_bal, args.objetivo)
    except ImportError:
        print("No se pudo encontrar el módulo de preprocesamiento")
        return
dicom_p = cli.subparsers.add_parser(
    "balance", help="Procesa archivos DICOM y genera CSV a partir de ellos."
)
dicom_p.add_argument("--ruta_csv",type=str,required=True, help="Ruta del dataset")
dicom_p.add_argument("--metodo_bal",type=str,required=True,choices=["smote", "smote-enn", "random-under"],help="Estrategia de balanceo.")
dicom_p.add_argument("--objetivo",type=str,default="Diagnostic",help="Clase objetivo para el balanceo (opcional). por defecto 'Diagnostic'")

dicom_p.set_defaults(func=process_dicom)
# fin integracion balanceo

# integracion vista-clsasificador
def process_dicom(args):
    from fedbiomed.modulo_dicom.vista import iniciar_interfaz

    iniciar_interfaz()
    """
    Función que procesa los archivos DICOM ubicados en el directorio indicado por el argumento '--path'.
    Se asume que existe un módulo 'mi_modulo_dicom' con la función 'process_dicom_directory'.
    """
    try:
        print("Iniciando interfaz gráfica para procesamiento DICOM...")
    except ImportError:
        print("No se pudo iniciar la interfaz gráfica")
        return
dicom_p = cli.subparsers.add_parser(
    "clasificador", help="Inicia la interfaz gráfica para procesamiento DICOM."
)

dicom_p.set_defaults(func=process_dicom)
# fin integracion vista-clasificador

def node(args):
    """Forwards node CLI"""
    NodeCLI = importlib.import_module("fedbiomed.node.cli").NodeCLI
    cli = NodeCLI()
    cli.parse_args(args)


def researcher(args):
    """Forwards researcher CLI"""
    ResearcherCLI = importlib.import_module("fedbiomed.researcher.cli").ResearcherCLI
    cli = ResearcherCLI()
    cli.parse_args(args)


node_p.set_defaults(func=node)
researcher_p.set_defaults(func=researcher)


def run():
    """Runs the CLI"""
    # This part executes know arguments
    args, extra = cli.parser.parse_known_args()
    # Forward arguments to Node or Researcher CLI
    if hasattr(args, "func") and args.func in [node, researcher]:
        args.func(extra)
    elif hasattr(args, "func"):
        args.func(args)
    # If there is no command provided
    else:
        cli.parse_args(["--help"])

if __name__ == "__main__":
        run()
