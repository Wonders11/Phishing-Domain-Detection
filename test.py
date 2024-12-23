from src.PhishingDomainDetection.pipelines.prediction_pipeline import PredictPipeline
    
pred = PredictPipeline.predict(5,2,4,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18)
print(pred)

# <html>
# <body>
# <form action="{{url_for('predict_datapoint')}}" method="POST">
#     <div class="form-group">
#         <label for="qty_slash_url">qty_slash_url:</label>
#         <input type="text" id="qty_slash_url" name="qty_slash_url" placeholder="Enter qty_slash_url value (float)" required>
#     </div>

#     <div class="form-group">
#         <label for="length_url">length_url:</label>
#         <input type="text" id="length_url" name="length_url" placeholder="Enter length_url value (float)" required>
#     </div>

#     <div class="form-group">
#         <label for="qty_dot_domain">qty_dot_domain:</label>
#         <input type="text" id="qty_dot_domain" name="qty_dot_domain" placeholder="Enter qty_dot_domain value (float)" required>
#     </div>

#     <div class="form-group">
#         <label for="qty_vowels_domain">qty_vowels_domain:</label>
#         <input type="text" id="qty_vowels_domain" name="qty_vowels_domain" placeholder="Enter qty_vowels_domain value (float)" required>
#     </div>

#     <div class="form-group">
#         <label for="domain_length">domain_length:</label>
#         <input type="text" id="domain_length" name="domain_length" placeholder="Enter domain_length value (float)" required>
#     </div>

#     <div class="form-group">
#         <label for="qty_dot_directory">qty_dot_directory:</label>
#         <input type="text" id="qty_dot_directory" name="qty_dot_directory" placeholder="Enter qty_dot_directory value (float)" required>
#     </div>

#     <div class="form-group">
#         <label for="qty_hyphen_directory">qty_hyphen_directory:</label>
#         <input type="text" id="qty_hyphen_directory" name="qty_hyphen_directory" placeholder="Enter qty_hyphen_directory value (float)" required>
#     </div>

#     <div class="form-group">
#         <label for="qty_underline_directory">qty_underline_directory:</label>
#         <input type="text" id="qty_underline_directory" name="qty_underline_directory" placeholder="Enter qty_underline_directory value (float)" required>
#     </div>

#     <div class="form-group">
#         <label for="file_length">file_length:</label>
#         <input type="text" id="file_length" name="file_length" placeholder="Enter file_length value (float)" required>
#     </div>

#     <div class="form-group">
#         <label for="time_response">time_response:</label>
#         <input type="text" id="time_response" name="time_response" placeholder="Enter time_response value (float)" required>
#     </div>

#     <div class="form-group">
#         <label for="asn_ip">asn_ip:</label>
#         <input type="text" id="asn_ip" name="asn_ip" placeholder="Enter asn_ip value (float)" required>
#     </div>

#     <div class="form-group">
#         <label for="time_domain_activation">time_domain_activation:</label>
#         <input type="text" id="time_domain_activation" name="time_domain_activation" placeholder="Enter time_domain_activation value (float)" required>
#     </div>

#     <div class="form-group">
#         <label for="time_domain_expiration">time_domain_expiration:</label>
#         <input type="text" id="time_domain_expiration" name="time_domain_expiration" placeholder="Enter time_domain_expiration value (float)" required>
#     </div>

#     <div class="form-group">
#         <label for="qty_nameservers">qty_nameservers:</label>
#         <input type="text" id="qty_nameservers" name="qty_nameservers" placeholder="Enter qty_nameservers value (float)" required>
#     </div>

#     <div class="form-group">
#         <label for="qty_mx_servers">qty_mx_servers:</label>
#         <input type="text" id="qty_mx_servers" name="qty_mx_servers" placeholder="Enter qty_mx_servers value (float)" required>
#     </div>

#     <div class="form-group">
#         <label for="ttl_hostname">ttl_hostname:</label>
#         <input type="text" id="ttl_hostname" name="ttl_hostname" placeholder="Enter ttl_hostname value (float)" required>
#     </div>

#     <div class="form-group">
#         <label for="tls_ssl_certificate">tls_ssl_certificate:</label>
#         <input type="text" id="tls_ssl_certificate" name="tls_ssl_certificate" placeholder="Enter tls_ssl_certificate value (float)" required>
#     </div>

#     <div class="form-group">
#         <label for="qty_redirects">qty_redirects:</label>
#         <input type="text" id="qty_redirects" name="qty_redirects" placeholder="Enter qty_redirects value (float)" required>
#     </div> 

#     <div style="clear:both;"></div>
#     <input type="submit" value="Submit">
# </form>
# </body>
# </html>








# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Prediction Results</title>
# </head>
# <body>
#     <h1> Given url is {{prediction_text}} </h1>
# </body>
# </html>