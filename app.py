import os
from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, request, session, url_for
from Model import Model

app = Flask(__name__)
app.secret_key = os.urandom(24)  # It's safer to use a random secret key

load_dotenv()
app.config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Ensure Model is initialized only if API key is present
if app.config['OPENAI_API_KEY']:
    model = Model(api_key=app.config['OPENAI_API_KEY'])

@app.route("/", methods=["GET", "POST"])
def index():
    """ Render index.html with chat history and token usage. """
    chat_history = session.get('chat_history', ['AI: Hello! I am a helpful chatbot that answer questions about cars!'])
    total_tokens = session.get("total_tokens", 0)
    price =  total_tokens / 1000000
    return render_template("index.html", total_tokens=total_tokens, price=price, chat_history=chat_history)

@app.route("/use", methods=["POST"])
def set_api():
    """ Handle API key form submission. """
    api_key = request.form.get("OPENAI_API_KEY")
    if api_key:
        app.config["OPENAI_API_KEY"] = api_key
        flash("Your API key has been saved successfully!", "success")
        global model
        model = Model(api_key=api_key)  # Reinitialize model with the new API key
    else:
        flash("Please enter a valid API key.", "error")
    return redirect(url_for("index"))

@app.route("/send_message", methods=["POST"])
def send_message():
    """ Handle message form submission and update session chat history. """
    message = request.form.get("input_message")
    if not message:
        flash("Your message cannot be empty.", "error")
        return redirect(url_for("index"))
    
    if 'OPENAI_API_KEY' not in app.config or app.config['OPENAI_API_KEY'] is None:
        flash("OPENAI_API_KEY is not set.", "error")
        return redirect(url_for("index"))

    # Process the message through the model
    model.invoke(message)
    # Update session state with latest from model
    session.modified = True  # Mark session as modified
    session['chat_history'] = [message.split(': ') for message in model.get_memory()['history'].split('\n')]
    session["total_tokens"] = model.get_tokens_count()

    return redirect(url_for("index"))  # Redirect to avoid form resubmission

if __name__ == "__main__":
    app.run(debug=True)
