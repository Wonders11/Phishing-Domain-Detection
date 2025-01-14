# Creation of WebAPI using Flask
# By default Flask runs on 5000 port number

from src.PhishingDomainDetection.pipelines.prediction_pipeline import PredictPipeline
from src.PhishingDomainDetection.logger import logging
# import inspect

from flask import Flask,request,render_template,jsonify,Response

# creating object of flask
app = Flask(__name__)

# creating routes
@app.route('/')
def home_page():
    return render_template("index.html")

# open browser and search localhost:8080 and it will show o/p

@app.route('/predict',methods=["GET","POST"])
def predict_datapoint(): # code for prediction will be here
    if request.method == "GET":
        return render_template("form.html")

    else:
        # data = {
        #         'qty_slash_url': float(request.form.get('qty_slash_url', 0)),
        #         'length_url': float(request.form.get('length_url', 0)),
        #         'qty_dot_domain': float(request.form.get('qty_dot_domain', 0)),
        #         'qty_vowels_domain': float(request.form.get('qty_vowels_domain', 0)),
        #         'domain_length': float(request.form.get('domain_length', 0)),
        #         'qty_dot_directory': float(request.form.get('qty_dot_directory', 0)),
        #         'qty_hyphen_directory': float(request.form.get('qty_hyphen_directory', 0)),
        #         'qty_underline_directory': float(request.form.get('qty_underline_directory', 0)),
        #         'file_length': float(request.form.get('file_length', 0)),
        #         'time_response': float(request.form.get('time_response', 0)),
        #         'asn_ip': float(request.form.get('asn_ip', 0)),
        #         'time_domain_activation': float(request.form.get('time_domain_activation', 0)),
        #         'time_domain_expiration': float(request.form.get('time_domain_expiration', 0)),
        #         'qty_nameservers': float(request.form.get('qty_nameservers', 0)),
        #         'qty_mx_servers': float(request.form.get('qty_mx_servers', 0)),
        #         'ttl_hostname': float(request.form.get('ttl_hostname', 0)),
        #         'tls_ssl_certificate': float(request.form.get('tls_ssl_certificate', 0)),
        #         'qty_redirects': float(request.form.get('qty_redirects', 0)),
        #     }
        
        # # Log the data for debugging
        # logging.info(f"Data received: {data}")

        qty_slash_url=float(request.form.get('qty_slash_url',0))
        length_url=float(request.form.get('length_url',0))
        qty_dot_domain=float(request.form.get('qty_dot_domain',0))
        qty_vowels_domain=float(request.form.get('qty_vowels_domain',0))
        domain_length=float(request.form.get('domain_length',0))
        qty_dot_directory=float(request.form.get('qty_dot_directory',0))
        qty_hyphen_directory=float(request.form.get('qty_hyphen_directory',0))
        qty_underline_directory=float(request.form.get('qty_underline_directory',0))
        file_length=float(request.form.get('file_length',0))
        time_response=float(request.form.get('time_response',0))
        asn_ip=float(request.form.get('asn_ip',0))
        time_domain_activation=float(request.form.get('time_domain_activation',0))
        time_domain_expiration=float(request.form.get('time_domain_expiration',0))
        qty_nameservers=float(request.form.get('qty_nameservers',0))
        qty_mx_servers=float(request.form.get('qty_mx_servers',0))
        ttl_hostname=float(request.form.get('ttl_hostname',0))
        tls_ssl_certificate=float(request.form.get('tls_ssl_certificate',0))
        qty_redirects=float(request.form.get('qty_redirects',0))

        # Log the received data for debugging
        logging.info(f"Received form data: {request.form.to_dict()}")

        # qty_slash_url=request.form.get('qty_slash_url')
        # length_url=request.form.get('length_url')
        # qty_dot_domain=request.form.get('qty_dot_domain')
        # qty_vowels_domain=request.form.get('qty_vowels_domain')
        # domain_length=request.form.get('domain_length')
        # qty_dot_directory=request.form.get('qty_dot_directory')
        # qty_hyphen_directory=request.form.get('qty_hyphen_directory')
        # qty_underline_directory=request.form.get('qty_underline_directory')
        # file_length=request.form.get('file_length')
        # time_response=request.form.get('time_response')
        # asn_ip=request.form.get('asn_ip')
        # time_domain_activation=request.form.get('time_domain_activation')
        # time_domain_expiration=request.form.get('time_domain_expiration')
        # qty_nameservers=request.form.get('qty_nameservers')
        # qty_mx_servers=request.form.get('qty_mx_servers')
        # ttl_hostname=request.form.get('ttl_hostname')
        # tls_ssl_certificate=request.form.get('tls_ssl_certificate')
        # qty_redirects=request.form.get('qty_redirects')

        logging.info('Prediction started')

        predict_pipeline = PredictPipeline()

        # print(inspect.signature(predict_pipeline.predict))

        pred = predict_pipeline.predict(qty_slash_url,
                                        length_url,
                                        qty_dot_domain,
                                        qty_vowels_domain,
                                        domain_length,
                                        qty_dot_directory,
                                        qty_hyphen_directory,
                                        qty_underline_directory,
                                        file_length,
                                        time_response,
                                        asn_ip,
                                        time_domain_activation,
                                        time_domain_expiration,
                                        qty_nameservers,
                                        qty_mx_servers,
                                        ttl_hostname,
                                        tls_ssl_certificate,
                                        qty_redirects
                                        )
        
        # r = Response(response=pred, status=200,mimetype='application/json')

        logging.info("Rendering results.html")

        # return render_template("results.html",prediction_text="{}".format(pred))
        return render_template("results.html", prediction_text=pred)



if __name__ ==  '__main__':
    app.run(host="0.0.0.0",port=8080) # without this we won't get the output

