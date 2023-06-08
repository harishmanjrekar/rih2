from flask import Flask, render_template, request
from transformers import BertForQuestionAnswering, BertTokenizer
import torch
import pyodbc
import openai

app = Flask(__name__)

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Connect to the database
mydb = pyodbc.connect(
    Driver="ODBC Driver 18 for SQL Server",
    Server="tcp:rih1.database.windows.net,1433",
    Database="rih",
    User="rih",
    Pwd="Adminpassword!",
    Encrypt="yes",
    TrustServerCertificate="no"
)

# Set up OpenAI API key
OPENAI_API_KEY = 'YOUR_OPENAI_API_KEY'
openai.api_key = OPENAI_API_KEY

def split_context(context, max_length):
    # Split a long context into smaller chunks of maximum length
    context_chunks = []
    words = context.split()
    chunk = ""
    for word in words:
        if len(chunk) + len(word) + 1 <= max_length:  # Add 1 for the space
            chunk += word + " "
        else:
            context_chunks.append(chunk.strip())
            chunk = word + " "
    context_chunks.append(chunk.strip())
    return context_chunks

def generate_answer(question, contexts):
    top_answer = None
    top_score = 0.0

    for context in contexts:
        if len(context) > 512:
            # Split long context into smaller chunks
            context_chunks = split_context(context, max_length=512)
            context_answers = []

            for chunk in context_chunks:
                inputs = tokenizer.encode_plus(question, chunk, return_tensors='pt', max_length=512, truncation=True)
                outputs = model(**inputs)
                start_scores = outputs.start_logits
                end_scores = outputs.end_logits
                start_index = torch.argmax(start_scores)
                end_index = torch.argmax(end_scores)
                input_ids = inputs['input_ids'].tolist()[0]
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                answer = tokenizer.decode(input_ids[start_index:end_index + 1])
                score = start_scores[0, start_index] + end_scores[0, end_index]
                context_answers.append((answer, score))

            # Select the best answer from all chunks of the context
            answer, score = max(context_answers, key=lambda x: x[1])
        else:
            inputs = tokenizer.encode_plus(question, context, return_tensors='pt', max_length=512, truncation=True)
            outputs = model(**inputs)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
            start_index = torch.argmax(start_scores)
            end_index = torch.argmax(end_scores)
            input_ids = inputs['input_ids'].tolist()[0]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            answer = tokenizer.decode(input_ids[start_index:end_index + 1])
            score = start_scores[0, start_index] + end_scores[0, end_index]

        if score > top_score:
            top_score = score
            top_answer = answer

    if (top_answer) and (top_score > 7):
        return [top_answer]
    else:
        return None

def get_response(prompt):
    response = openai.Completion.create(
        engine="text-curie-001",
        prompt=prompt,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.4,
    )
    return response.choices[0].text.strip()

def save_conversation(question, answer):
    mycursor = mydb.cursor()
    sql = "INSERT INTO conversations (question, answer) VALUES (?, ?)"
    val = (question, answer)
    mycursor.execute(sql, val)
    mydb.commit()
    print("Conversation saved to database")

@app.route('/', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        question = request.form['question']
        answer = request.form['answer']

        if question and answer:
            save_conversation(question, answer)

    return render_template('chatbot.html')

@app.route('/get-response', methods=['POST'])
def get_chatbot_response():
    question = request.form['question']
    answer = None

    # Fetch contexts from the Azure SQL database
    cursor = mydb.cursor()
    cursor.execute("SELECT answer FROM conversations")
    contexts = [row[0] for row in cursor.fetchall()]

    answers = generate_answer(question, contexts)
    if answers:
        answer = answers[0]
    else:
        answer = get_response(question)
        save_conversation(question, answer)

    return {'answer': answer}

if __name__ == '__main__':
    app.run(debug=True)
