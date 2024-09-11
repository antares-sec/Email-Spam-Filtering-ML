import pickle


class EmailSpammner:
    
    def __init__(self, message):
        self.message = [message]
        self.model_cls = "email_spam_v2.pkl" # Model for email spam
        self.vectorizer = "vectorizer_v2.pkl" # Vectorizer, it'll used for vectorizer the text input


    def vectorizerMessage(self):
        with open(self.vectorizer, 'rb') as file:
            model = pickle.load(file)

        # Change the message into numeric features (2D-Array)
        message = model.transform(self.message).toarray()
        return message
    
    def predictor(self):
        with open(self.model_cls, 'rb') as file:
            model = pickle.load(file)

        # Getting the final message
        email = self.vectorizerMessage()
        # Predict the email
        pred = model.predict(email)

        # Check if 1 is a spam and 0 is not a spam
        if pred == 1:
            print("The message is a spam!")
        elif pred == 0:
            print("The message not a spam!")
            
    def run(self):
        self.predictor()


if __name__ == '__main__':
    email = EmailSpammer('Congrats! You won a $1000. Please follow this step to get your money! [Link]')
    email.run()
