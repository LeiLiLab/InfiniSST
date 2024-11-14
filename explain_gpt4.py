import pandas as pd
from tqdm import tqdm
from openai import OpenAI
df = pd.read_csv('example_generation.csv')
with open('/mnt/aries/home/siqiouyang/.openai/api_keys/siqi/api_key_1', 'r') as r:
    api_key = r.read()
client = OpenAI(api_key=api_key)
model = 'gpt-4'
prompt = """English source: I do apologise about this, we must gain permission from the account holder to discuss
an order with another person, I apologise if this was done previously, however, I would not be able
to discuss this with yourself without the account holders permission.
German translation: Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung
mit einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen wäre, aber
ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir involvement.
MQM annotations:
Critical:
no-error
Major:
accuracy/mistranslation - "involvement"
accuracy/omission - "the account holder"
Minor:
fluency/grammar - "wäre"
fluency/register - "dir"
English source: Talks have resumed in Vienna to try to revive the nuclear pact, with both sides
trying to gauge the prospects of success after the latest exchanges in the stop-start negotiations.
Czech translation: Ve Vídni se ve Vídni obnovily rozhovory o oživení jaderného paktu, přičemže obě
partaje se snaží posoudit vyhlídky na úspěch po posledních výměnách v jednáních.
MQM annotations:
Critical:
no-error
Major:
accuracy/addition - "ve Vídni"
accuracy/omission - "the stop-start"
Minor:
terminology/inappropriate for context - "partaje"
Chinese source: 大众点评乌鲁木齐家居商场频道为您提供高铁居然之家地址，电话，营业时间等最新商户信息，
找装修公司，就上大众点评
English translation: Urumqi Home Furnishing Store Channel provides you with the latest business
information such as the address, telephone number, business hours, etc., of high-speed rail, and
find a decoration company, and go to the reviews.
MQM annotations:
Critical:
accuracy/addition - "of high-speed rail"
Major:
accuracy/mistranslation - "go to the reviews"
Minor:
style/awkward - "etc.,"
English source: {}
Spanish translation: {}
MQM annotations:
"""
explanations = []
for i in tqdm(range(100)):
    text = prompt.format(df['src'][i], df['hyp_bi'][i])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
        ],
        temperature=0
    )

    bi_result = response.choices[0].message.content

    text = prompt.format(df['src'][i], df['hyp_uni_waco'][i])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
        ],
        temperature=0
    )

    uni_result = response.choices[0].message.content

    with open('analysis/{}.txt'.format(i), 'w') as w:
        w.write('English source: {}\n\n'.format(df['src'][i]))

        w.write('Bi  translation: {}\n{}\n\n'.format(df['hyp_bi'][i], bi_result))
        w.write('Uni translation: {}\n{}\n\n'.format(df['hyp_uni_waco'][i], uni_result))