from service.TaxiFaresPredictionNYC import TaxiFaresPredictionNYC
from service.Preprocess import Preprocess
import config


class Service:

    @staticmethod
    def get_service(service_type):
        if service_type == "NN":
            return TaxiFaresPredictionNYC()
        elif service_type == "Preprocess":
            return Preprocess()

        return None
