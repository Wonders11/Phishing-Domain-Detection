# qty_slash_url,length_url,qty_dot_domain,qty_vowels_domain,domain_length,qty_dot_directory,
# qty_hyphen_directory,qty_underline_directory,file_length,time_response,asn_ip,time_domain_activation,
# time_domain_expiration,qty_nameservers,qty_mx_servers,ttl_hostname,tls_ssl_certificate,qty_redirects

import os
import sys
import numpy as np
import pandas as pd
from src.PhishingDomainDetection.exception import customexception
from src.PhishingDomainDetection.logger import logging
from src.PhishingDomainDetection.utils.utils import load_object # used for loading physical object (pickle files)

class PredictPipeline:

    def __init__(self):
        # Initialize paths for preprocessor and model
        print("init.. the object")

    def predict(qty_slash_url,length_url,qty_dot_domain,qty_vowels_domain,domain_length,qty_dot_directory,qty_hyphen_directory,
                qty_underline_directory,file_length,time_response,asn_ip,time_domain_activation,time_domain_expiration,qty_nameservers,
                qty_mx_servers,ttl_hostname,tls_ssl_certificate,qty_redirects):
        try:

            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            columns = ["qty_slash_url", "length_url", "qty_dot_domain", "qty_vowels_domain", "domain_length", 
                       "qty_dot_directory", "qty_hyphen_directory", "qty_underline_directory", "file_length", 
                       "time_response", "asn_ip", "time_domain_activation", "time_domain_expiration", 
                       "qty_nameservers", "qty_mx_servers", "ttl_hostname", "tls_ssl_certificate", "qty_redirects"]
            
            data = [[qty_slash_url,length_url,qty_dot_domain,qty_vowels_domain,domain_length,qty_dot_directory,qty_hyphen_directory,
                qty_underline_directory,file_length,time_response,asn_ip,time_domain_activation,time_domain_expiration,qty_nameservers,
                qty_mx_servers,ttl_hostname,tls_ssl_certificate,qty_redirects]]
            
            features = pd.DataFrame(data,columns=columns)

            scaled_fea=preprocessor.transform(features)
            
            pred=model.predict(scaled_fea)

            pred1 = "malicious" if pred==1 else "legitimate"

            return pred1

        except Exception as e:
            raise customexception(e,sys)


class CustomData:
    def __init__(self,
                qty_slash_url:float,
                length_url:float,
                qty_dot_domain:float,
                qty_vowels_domain:float,
                domain_length:float,
                qty_dot_directory:float,
                qty_hyphen_directory:float,
                qty_underline_directory:float,
                file_length:float,
                time_response:float,
                asn_ip:float,
                time_domain_activation:float,
                time_domain_expiration:float,
                qty_nameservers:float,
                qty_mx_servers:float,
                ttl_hostname:float,
                tls_ssl_certificate:float,
                qty_redirects:float
                ):
        
        self.qty_slash_url=qty_slash_url
        self.length_url=length_url
        self.qty_dot_domain=qty_dot_domain
        self.qty_vowels_domain=qty_vowels_domain
        self.domain_length=domain_length
        self.qty_dot_directory=qty_dot_directory
        self.qty_hyphen_directory = qty_hyphen_directory
        self.qty_underline_directory = qty_underline_directory
        self.file_length = file_length
        self.time_response = time_response
        self.asn_ip = asn_ip
        self.time_domain_activation = time_domain_activation
        self.time_domain_expiration = time_domain_expiration
        self.qty_nameservers = qty_nameservers
        self.qty_mx_servers = qty_mx_servers
        self.ttl_hostname = ttl_hostname
        self.tls_ssl_certificate = tls_ssl_certificate
        self.qty_redirects = qty_redirects
            
    def get_data_as_dataframe(self):

        try:
            # Input which will be given to predict
            custom_data_input_dict = {
                'qty_slash_url':[self.qty_slash_url],
                'length_url':[self.length_url],
                'qty_dot_domain':[self.qty_dot_domain],
                'qty_vowels_domain':[self.qty_vowels_domain],
                'domain_length':[self.domain_length],
                'qty_dot_directory':[self.qty_dot_directory],
                'qty_hyphen_directory':[self.qty_hyphen_directory], 
                'qty_underline_directory':[self.qty_underline_directory], 
                'file_length': [self.file_length], 
                'time_response':[self.time_response], 
                'asn_ip':[self.asn_ip], 
                'time_domain_activation':[self.time_domain_activation], 
                'time_domain_expiration':[self.time_domain_expiration], 
                'qty_nameservers':[self.qty_nameservers],
                'qty_mx_servers':[self.qty_mx_servers],
                'ttl_hostname':[self.ttl_hostname],
                'tls_ssl_certificate':[self.tls_ssl_certificate], 
                'qty_redirects':[self.qty_redirects]
                }
            # converting dictionary to dataframe
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise customexception(e,sys)