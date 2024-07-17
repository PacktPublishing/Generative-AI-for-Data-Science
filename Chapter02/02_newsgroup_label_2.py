

from langchain_core.prompts import PromptTemplate
import random, os

import pickle 
import numpy as np
from sklearn.metrics import f1_score
import json


from langchain_google_genai import ChatGoogleGenerativeAI
from getpass import getpass

# set up Google for serving Gemini LLM
if not os.environ.get('GOOGLE_API_KEY'):
      os.environ['GOOGLE_API_KEY'] =  getpass(prompt="please paste your Google API key: ")
BASE_LLM = ChatGoogleGenerativeAI
base_llm_model = "gemini-1.5-flash"

# for Gemma model  as Option B
#from langchain_community.llms import Ollama # install ollama locally
#BASE_LLM = Ollama
#base_llm_model = "gemma"

# or Option C: use ChatOpenAI and gpt-4

# create a prompt template
template = """Given this list of labels as json:
```
    {topics}
```
Return which label would apply to the following article:
```
    {article}
```
Return just the label for the topic. do not bold or apply markup
"""

prompt_template = PromptTemplate(
    input_variables=["topics", "article"],
    template=template
)

# load the saved data 
data = pickle.load(open("saved.pk", "rb"))
X_test = data['X_test']
y_test = data['y_test']
f1_score_rf = data['f1_score_rf']

topics =  ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 
     'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 
     'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 
     'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 
     'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

enumerated_topics = list(enumerate(topics))

# create the llm instance
llm = BASE_LLM(
    model=base_llm_model,
    temperature=.9,
    verbose=True
)  


count = 0
out = []
# score the test data with the LLM
for article in X_test:

    # format the prompt to add variable values
    prompt_formatted_str: str = prompt_template.format(
        article=article,
        topics=json.dumps(topics))
    score = llm.invoke(prompt_formatted_str)
    if hasattr(score, "content"):
         score = score.content
    try:
        score = topics.index(score.strip())
    except ValueError:
        print(f"could not find '{score}' filling in a random choice")
        score = topics.index(random.choice(topics))
    expected = y_test[count]
    count += 1
    print(f"#{count}: {score}/{expected}")
    out.append(int(score))

y_pred = np.array(out)

f1 = f1_score(y_test, y_pred, average='macro')
print('F1 score:', f1)

# F1 score: 0.6705069107660846