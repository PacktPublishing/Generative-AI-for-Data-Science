import os
import pickle 
import json
from google.cloud import storage
from getpass import getpass
import vertexai
from vertexai.preview.tuning import sft


def save_dicts_to_gcs(dict_list, bucket_name, destination_blob_name, json_file_name):
    """
    Save a list of dictionaries to a JSON file and upload it to a GCS bucket.

    :param dict_list: List of dictionaries to save.
    :param bucket_name: Name of the GCS bucket.
    :param destination_blob_name: The name of the destination blob (including the path in the bucket).
    :param json_file_name: The name of the JSON file to create.
    """

    # Save JSON data to a file
    with open(json_file_name, 'w+') as json_file:
        for line in dict_list:
            json_file.write(json.dumps(line) + "\n")

    # Initialize a GCS client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.get_bucket(bucket_name)

    # Create a blob in the bucket
    blob = bucket.blob(destination_blob_name)

    # Upload the file to GCS
    blob.upload_from_filename(json_file_name)

    print(f"File {json_file_name} uploaded to {bucket_name}/{destination_blob_name}.")

    return f"gs://{bucket_name}/{destination_blob_name}"


# set up Google for serving Gemini LLM
if not os.environ.get('GOOGLE_API_KEY'):
      os.environ['GOOGLE_API_KEY'] =  getpass(prompt="please paste your Google API key: ")

# create a prompt template

topics =  ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 
     'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 
     'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 
     'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 
     'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

# load the saved data 
data = pickle.load(open("saved.pk", "rb"))

X_train = data['X_train']
y_train = data['y_train']

out = []
for i in range(len(X_train)):
     article = X_train[i]
     label = topics[y_train[i]]
     out.append({
        "messages": [
        {
            "role": "system",
            "content": f"You should classify the text into one of the following classes: {topics}"
        },
        { "role": "user", "content": article },
        { "role": "model", "content": label }
        ]
    })


# save training to GCS

train_dataset = save_dicts_to_gcs(out, "caltech_newsgroup_classify", 
                  "newsgroup_train.jsonl", "newsgroup_train.jsonl")

if not os.environ.get('PROJECT_ID'):
    os.environ['PROJECT_ID'] =  getpass(prompt="please paste your Google PROJECT ID: ")

if not os.environ.get('PROJECT_REGION'):
    os.environ['PROJECT_REGION'] =  getpass(prompt="please paste your Google PROJECT Region: ")
     
vertexai.init(project=os.environ['PROJECT_ID'], 
              location=os.environ['PROJECT_REGION'])

sft_tuning_job = sft.train(
    source_model="gemini-1.0-pro-002",
    train_dataset=train_dataset,
    # The following parameters are optional
   # validation_dataset="gs://cloud-samples-data/ai-platform/generative_ai/sft_validation_data.jsonl",
    epochs=2,
    adapter_size=1,
    learning_rate_multiplier=1.0,
    tuned_model_display_name="newsgroup_tuned",
)

i = 1




