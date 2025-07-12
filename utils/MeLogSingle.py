"""Nova Classe de Log - Singleton"""

import configparser
import logging
from datetime import datetime


class MeLogger(logging.Logger):
    """Extensão da Classe Logger, customizada para a aplicação"""

    _instance = None

    # ---------------------------------------------------------------------------------------------
    def __new__(cls, *args, **kwargs):
        """Força a ter apenas uma instância Design pattern Singleton
        Por convenção estou mantendo os mesmos nomes dos parâmentos e não
        usando o type hint
        """

        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    # ---------------------------------------------------------------------------------------------
    def __init__(self):
        # O nome da primeira instância é Looger então troca-se o nome para não
        # precisar fazer todo o init novamente.

        if self.__class__.__name__ != "Logger_Criado":

            # usando um nível de debug mais abrangente fixo para ter liberdade de uso.
            level = "DEBUG"
            
            # tem que passar pelo init do pai para criar o logger efetivamente.
            super().__init__(self.__class__.__name__, level=level)
            
            # troca o nome da classe padrão para não executar mais de uma vez.
            self.__class__.__name__ = "Logger_Criado"

            self.setLevel(level)

            formatter = logging.Formatter(
                "%(asctime)s %(filename)s %(levelname)s - Line: %(lineno)02d - %(message)s ",
                datefmt="%d-%m-%y %H:%M:%S",
            )

            # Informações para o Manipulador de Arquivos

            file_handler = logging.FileHandler(self._monta_nome_arquivo())
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)

            # Informações para impressão na Tela
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(level)
            stream_handler.setFormatter(formatter)

            self.addHandler(file_handler)
            self.addHandler(stream_handler)

    # ---------------------------------------------------------------------------------------------
    @staticmethod
    def _monta_nome_arquivo() -> str:
        """Montagem do nome do arquivo"""

        config = configparser.ConfigParser()
        config.read("./utils/app.ini")

        path_log = config.get("log", "path_log")
        natureza = config.get("log", "natureza")

        inicio = datetime.now()

        nome = f"{path_log}{natureza.replace(' ', '_')}_\
                 {inicio.year}{str(inicio.month).zfill(2)}\
                 {str(inicio.day).zfill(2)}.log".replace(
            " ", ""
        )

        return nome