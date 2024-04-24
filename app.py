from flask import Flask, render_template, url_for, request
import os
import zipfile
from running_model import run_model_on_dataset

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        filesDict = request.files.to_dict()
        print('files dict:', filesDict)
        uploadData=request.files['media']
        print('upload data:', uploadData)
        data_file_name = uploadData.filename
        folder_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
        file_path = os.path.join(folder_path, data_file_name)
        uploadData.save(file_path)

        # Extract the uploaded zip file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(folder_path, 'model'))

        # Get the list of files extracted from the zip file
        extracted_files = zip_ref.namelist() 
        if len(extracted_files) > 1:
            extracted_file = extracted_files[0].split('/')[0]
        else:
            extracted_file = extracted_files[0]

        # Run the model on the dataset
        accuracy = run_model_on_dataset(os.path.join(folder_path, 'model', extracted_file))

        return render_template('result.html', accuracy=accuracy)
        
        # return "File has been uploaded."
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)