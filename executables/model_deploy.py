from flask import Flask, request, jsonify,render_template
import pickle

# Initialize Flask app
app = Flask(__name__)

try:
  from joblib import load
  model = load('model_reg.pkl')
except ImportError:
  # If joblib fails, use pickle.load
  try:
    model = pickle.load(open('model.pkl', 'rb'))
  except FileNotFoundError:
    print("Error: Model file not found!")


@app.route('/')
def home():
  """Renders the home page with a prediction button."""
  return render_template("HOME_m.html")

@app.route('/predict') 
def index():
    return render_template("index1.html")

@app.route('/data_predict', methods=['POST'])

def predict():
    try:
        # Print incoming form data for debugging
        print("Form data received:", request.form)
        """Handles user input, makes prediction, and returns JSON response."""
  
    # Extract user input from form data
        age = float(request.form['age'])
        gendermale = float(request.form['gendermale'])
        genderfemale = float(request.form['genderfemale'])
        tb = float(request.form['tb'])
    #db = float(request.form['db'])
        ap = float(request.form['ap'])
        aa1 = float(request.form['aa1'])
    #aa2 = float(request.form['aa2'])
        tp = float(request.form['tp'])
        a = float(request.form['a'])
        agr = float(request.form['agr'])

    # Prepare input data
        data = [[age, tb, ap, aa1, tp, a, agr, genderfemale, gendermale]]

    # Make prediction
        prediction = float(model.predict(data)[0])
        if prediction==1.0:
            res="you have a liver disease problem"
        else:
            res="you do not have a liver disease"

    # Prepare JSON response
        return render_template('index1.html', x=res)

    except Exception as e:
        
    # Handle errors gracefully (e.g., invalid input, model loading issues)
       return jsonify({'error': str(e)}), 400  # Bad request status code

if __name__ == '__main__':
  app.run(debug=False)  # Set debug=False for production
