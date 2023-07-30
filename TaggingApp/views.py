from django.shortcuts import render

from joblib import load

model =load('./savedModels/model.joblib')

from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", max_length=512)
nlp= pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Create your views here.
def predictor(request):
    if request.method == 'POST':
        news_text = request.POST['news_text']
        y_pred = nlp([news_text])
        final =y_pred[0]['label']
        return render(request,'main.html',{'result' :final})
    return render(request,'main.html')



    