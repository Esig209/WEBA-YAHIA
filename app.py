from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask import render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat.db'
db = SQLAlchemy(app)


class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_message = db.Column(db.String(500))
    bot_response = db.Column(db.String(500))
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())


@app.route("/")
def index():
    return render_template("chat.html")


@app.route('/get', methods=['POST'])
def get_bot_response():
    user_message = request.form['message']
    bot_response = get_Chat_response(user_message)
    try:
        new_conversation = Conversation(user_message=user_message, bot_response=bot_response)
        db.session.add(new_conversation)
        db.session.commit()
        return jsonify({'response': bot_response})
    except Exception as e:
        db.session.rollback()  # Important pour annuler la transaction en cas d'erreur
        print("Une erreur est survenue : ", e)
        # Vous pouvez choisir de renvoyer un message d'erreur à l'utilisateur
        return jsonify({'error': 'Une erreur interne est survenue. Veuillez réessayer.'}), 500


@app.route("/conversations", methods=["GET"])
def show_conversations():
    conversations = Conversation.query.all()
    return jsonify([{"user_message": conv.user_message, "bot_response": conv.bot_response} for conv in conversations])


@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        db.session.query(Conversation).delete()
        db.session.commit()
        return 'Historique effacé avec succès.'
    except Exception as e:
        db.session.rollback()
        print("Une erreur est survenue lors de la tentative d'effacement de l'historique : ", e)
        return 'Une erreur est survenue lors de la tentative d\'effacement de l\'historique.'


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)


def get_Chat_response(text):
    for step in range(5):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens,
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # pretty print last ouput tokens from bot
        return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)


if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
